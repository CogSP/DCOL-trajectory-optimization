import numpy as np
import h5py
import copy
from primitives.misc_primitive_constructor import SphereMRP, create_rect_prism, create_n_sided, CylinderMRP, CapsuleMRP, ConeMRP, PolytopeMRP, PolygonMRP
from proximity.proximity import proximity_mrp
from proximity.proximity_gradient import proximity_gradient
from primitives.problem_matrices import dcm_from_mrp

def skew(w):
    """
    Skew-symmetric matrix from a 3D vector.

    """
    return np.array([[0, -w[2], w[1]],
                     [w[2], 0, -w[0]],
                     [-w[1], w[0], 0]])


def dynamics(params, x, u, k_iter):
    """
    Compute the dynamics of the quadrotor using MRP for attitude representation.
    
    Args:
        params: NamedTuple-like dictionary with simulation parameters.
        x: State vector [position (3), velocity (3), MRP (3), angular velocity (3)].
        u: Control inputs (rotor speeds).
        k_iter: Iteration index (unused).
    
    Returns:
        np.ndarray: Derivative of the state vector.
    """
    # Extract state variables
    r = x[:3]        # Position
    v = x[3:6]       # Velocity
    p = x[6:9]       # Attitude (MRP)
    omega = x[9:12]  # Angular velocity

    # Direction cosine matrix from MRP
    Q = dcm_from_mrp(p)  # Assume dcm_from_mrp function is implemented.

    # Physical constants
    mass = 0.5
    J = np.diag([0.0023, 0.0023, 0.004])  # Inertia matrix
    gravity = np.array([0, 0, -9.81])
    L = 0.1750
    kf = 1.0
    km = 0.0245

    # Rotor speeds
    w1, w2, w3, w4 = u

    # Rotor forces (positive values only)
    F1 = max(0, kf * w1)
    F2 = max(0, kf * w2)
    F3 = max(0, kf * w3)
    F4 = max(0, kf * w4)

    # Total rotor force in body frame
    F = np.array([0., 0., F1 + F2 + F3 + F4])

    # Rotor torques
    M1 = km * w1
    M2 = km * w2
    M3 = km * w3
    M4 = km * w4
    tau = np.array([L * (F2 - F4), L * (F3 - F1), (M1 - M2 + M3 - M4)])  # Total torque

    # Forces in the world frame
    f_world = mass * gravity + Q @ F

    # Angular velocity dynamics
    I = np.eye(3)
    p_norm_sq = np.dot(p, p)
    p_term = ((1 + p_norm_sq) / 4) * (I + 2 * (np.dot(skew(p), skew(p)) + skew(p)) / (1 + p_norm_sq))
    angular_dynamics = np.linalg.solve(J, tau - np.cross(omega, J @ omega))

    # State derivatives
    state_dot = np.concatenate([
        v,
        f_world / mass,
        np.dot(p_term, omega),
        angular_dynamics
    ])
    return state_dot

def discrete_dynamics(params, x, u, k):
    """
    Simulates system dynamics using RK4 integration.

    Args:
        params (dict): Simulation parameters.
        x (ndarray): Current state.
        u (ndarray): Control input.
        k (int): Current timestep index.

    Returns:
        ndarray: Updated state after integration.
    """
    dt = params['dt']
    k1 = dt * dynamics(params, x, u, k)
    k2 = dt * dynamics(params, x + 0.5 * k1, u, k)
    k3 = dt * dynamics(params, x + 0.5 * k2, u, k)
    k4 = dt * dynamics(params, x + k3, u, k)

    return x + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

def inequality_constraints_u(params, u):
    """
    Inequality constraints for the control inputs.

    """
    u_max = params['u_max']
    u_min = params['u_min']
    return np.concatenate([u - u_max, -u + u_min])

def inequality_constraints_x(params, x):
    """
    Calculates state-based inequality constraints.

    Args:
        params (dict): Simulation parameters.
        x (ndarray): Current state.

    Returns:
        ndarray: Constraint violations for the current state.
    """
    params['P_vic'].r = np.array(x[0:3])
    params['P_vic'].p = np.array(x[6:9])
    
    contacts = []
    for obs in params['P_obs']:
        alpha, _ = proximity_mrp(params['P_vic'], obs, verbose=False)
        contacts.append(1 - alpha)
        
    return np.array(contacts)
    
import numpy as np

def inequality_constraints_x_grad(params, x):
    """
    Compute the Jacobian matrix for state inequality constraints.

    Args:
        params (dict): Dictionary containing the environment parameters, including P_vic and P_obs.
        x (numpy.ndarray): State vector.

    Returns:
        numpy.ndarray: Stacked Jacobian matrix for all constraints.
    """
    # Update P_vic's position and orientation based on the state vector
    params["P_vic"].r = np.array(x[:3])  # Position
    params["P_vic"].p = np.array(x[6:9])  # Orientation (Modified Rodrigues Parameters, MRP)

    # Calculate the Jacobians from the DCOL library
    Js = [proximity_gradient(params["P_vic"], obs)[1] for obs in params["P_obs"]]


    # Extract relevant portions of the Jacobians for each constraint
    contact_J = [
        np.hstack([
            -Js[i][:3].reshape(1, -1),  
            np.zeros((1, 3)),           
            -Js[i][3:6].reshape(1, -1), 
            np.zeros((1, 3))            
        ])
        for i in range(len(params["P_obs"]))
    ]


    # Stack all Jacobians vertically
    return np.vstack(contact_J)



def inequality_constraints_u_grad(params, u):
    """
    Compute the Jacobian matrix for control inequality constraints.

    Args:
        params (dict): Dictionary containing the environment parameters, including control dimensions.
        u (numpy.ndarray): Control vector.

    Returns:
        numpy.ndarray: Stacked Jacobian matrix for all control constraints.
    """
    nu = params["nu"]
    identity = np.eye(nu)
    return np.vstack([identity, -identity])



def linear_interp(dt, x0, xg, N):
    """
    Linearly interpolates between two points x0 and xg over N points.

    Args:
        dt (float): Time step.
        x0 (numpy.ndarray): Initial state.
        xg (numpy.ndarray): Goal state.
        N (int): Number of points to interpolate.

    Returns:
        numpy.ndarray: Interpolated trajectory.
    """

    delta_p = (xg[0:3] - x0[0:3])
    positions = np.array([((i-1)*(delta_p/(N-1)) + x0[0:3]) for i in range(1, N+1)])
    delta_p = (xg[6:9] - x0[6:9])
    attitudes = np.array([((i-1)*(delta_p/(N-1)) + x0[6:9]) for i in range(1, N+1)])

    # Assertions to verify the interpolation
    assert np.array_equal(positions[0], x0[0:3]), "Initial position does not match x0[0:3]"
    assert np.array_equal(positions[-1], xg[0:3]), "Final position does not match xg[0:3]"
    assert np.array_equal(attitudes[0], x0[6:9]), "Initial attitude does not match x0[6:9]"
    assert np.array_equal(attitudes[-1], xg[6:9]), "Final attitude does not match xg[6:9]"


    velocities = [delta_p/((N-1)*dt) for i in range(N)]
    angular_velocities = np.zeros(3)

    trajectory = [
        np.concatenate([positions[i], velocities[i], attitudes[i], angular_velocities]) for i in range(N)
    ]

    return trajectory

def initialize_quadrotor():
    """
    Initializes the parameters for the quadrotor system.

    """

    print("Initializing parameters for the quadrotor...")
    nx = 12
    nu = 4
    N = 100
    dt = 0.08
    max_linesearch_iters = 20
    atol = 1e-2
    max_iters = 3000
    reg_min = 1e-6
    reg = reg_min
    reg_max = 1e2
    rho=1e0 
    phi=10.0
    convio_tol = 1e-4


    X_hist = []
    U_hist = []
    hx_hist = []
    hu_hist = []

    # the state is [r;v;p;ω]
    x0 = np.array([-8, 0, 4, 0, 0, 0.0, 0, 0, 0, 0, 0, 0])
    xg = np.array([8, 0, 4, 0, 0, 0.0, 0, 0, 0, 0, 0, 0])

    Xref = linear_interp(dt,x0,xg,N)

    # F = m*g/4 since nu = 4 is the number of rotors
    # force divided equally among the rotors to counteract gravity
    Uref = [(9.81*0.5/4)*np.ones(nu) for i in range(N)]

    Q = np.diag(np.ones(nx))
    Qf = np.diag(np.ones(nx))
    R = np.diag(np.ones(nu))

    P_vic = SphereMRP(radius=0.25)

    # Path to your .jld2 file
    file_path = "systems/polytopes.jld2"

    # Open the file using h5py
    with h5py.File(file_path, "r") as f:
        # Access datasets
        A1 = f["A1"][:]
        b1 = f["b1"][:]
        A2 = f["A2"][:]
        b2 = f["b2"][:]

    # ground
    P_bot = create_rect_prism(length=20, width=5, height=0.2)
    P_bot.r = [0, 0, 0.9]
    
    # ceiling
    P_top = create_rect_prism(length=20, width=5, height=0.2)
    P_top.r = [0, 0, 6.0]


    dictionary = create_n_sided(5, 0.6)
    
    A_polygon = dictionary["A"]
    B_polygon = dictionary["b"]

    P_obs = [
        CylinderMRP(radius=0.6, height=3.0), 
        CapsuleMRP(radius=0.2, height=5.0),
        SphereMRP(radius=0.8),
        ConeMRP(height=2.0, beta=np.deg2rad(22)),
        PolytopeMRP(A2.T, b2),
        PolygonMRP(A_polygon, B_polygon, 0.2),
        CylinderMRP(radius=1.1, height=2.3),
        CapsuleMRP(radius=0.8, height=1.0),
        SphereMRP(radius=0.5),
        P_bot,
        P_top
    ]


    # IMPORTANT NOTE: The following values are hardcoded for the initial positions and orientations of the obstacles.
    # In the original Julia code implementation, these values were obtained "randomly" with seeed value 2. 
    # Since the same seed will not produce the same values in Python, we just took the values from the original implementation
    # and put them here to have the same experiment and results 
    P_obs[0].r = [-5.0, -0.3597289068234817, 4.087208492428585]
    P_obs[0].p = [0.9743462834661368, 0.5695654691654629, -0.929297065594203]
    P_obs[1].r = [-3.75, 2.0547630560640364, 3.3248927294469155]
    P_obs[1].p = [0.44432216225861665, -0.8131633664490159, 0.8533462452863487]
    P_obs[2].r = [-2.5, 0.01357380155160959, 3.1056516058837307]
    P_obs[2].p = [-0.7818142467739891, -1.0606493186561021, -0.6997594248738506]
    P_obs[3].r = [-1.25, 0.1520302408349855, 2.100626290031169]
    P_obs[3].p = [0.09970204047057568, -0.6590733218999884, 0.10747184882042882]
    P_obs[4].r = [0.0, 0.27038613194550204, 4.579317307027433]
    P_obs[4].p = [-1.178486073522902, -0.5852806292416908, -0.5104503832374265]
    P_obs[5].r = [1.25, -0.20563037602802728, 3.7707031750912097]
    P_obs[5].p = [1.322242556684692, 1.477962368008582, -0.09186250030835676]
    P_obs[6].r = [2.5, 1.724189934074888, 3.1527083547286816]
    P_obs[6].p = [-1.670756785490579, -1.6504683581003534, 0.9958143390876766]
    P_obs[7].r = [3.75, -0.7885513165549604, 2.3533371368422706]
    P_obs[7].p = [0.40980738483268503, 0.5108420391824778, 0.42272633604120335]
    P_obs[8].r = [5.0, 0.32074771862886275, 4.251199978479224]
    P_obs[8].p = [1.8822143307659809, -0.7779808480817001, 0.8308676764061569]


    # control bounds
    u_min = -2000*np.ones(nu)
    u_max =  2000*np.ones(nu)
    ncx = len(P_obs)
    ncu = 2*nu

    params = {
        "nx": nx,
        "nu": nu,
        "ncx": ncx,
        "ncu": ncu,
        "N": N,
        "Q": Q,
        "R": R,
        "Qf": Qf,
        "u_min": u_min,
        "u_max": u_max,
        "Xref": Xref,
        "Uref": Uref,
        "dt": dt,
        "P_obs": P_obs,
        "P_vic": P_vic,
        'max_linesearch_iters': max_linesearch_iters,
        'atol': atol,
        'max_iters': max_iters,
        'reg_min': reg_min,
        'reg': reg,
        'reg_max': reg_max,
        'rho': rho,
        'phi': phi,
        'convio_tol': convio_tol,
        'system': 'quadrotor',
        'X_hist': X_hist,
        'U_hist': U_hist,
        'hx_hist': hx_hist,
        'hu_hist': hu_hist,
    }

    X = [copy.deepcopy(x0) for i in range(N)]


    # IMPORTANT NOTE: The following values are hardcoded for the initial positions and orientations of the obstacles.
    # In the original Julia code implementation, these values were obtained "randomly" with seeed value 2. 
    # Since the same seed will not produce the same values in Python, we just took the values from the original implementation
    # and put them here to have the same experiment and results
    U = np.array([[1.2263980220524084, 1.2263388518645655, 1.2263445368320691, 1.2263057090533414], [1.2263203973047678, 1.2261688260326638, 1.2262802097008272, 1.2263294374052953], [1.2263360267052048, 1.2261832519533125, 1.226366842535274, 1.22624957935308], [1.2262796340799025, 1.2263305852618474, 1.2262404851233923, 1.226323320505437], [1.2263500845416275, 1.2262618087636128, 1.2261830791804376, 1.2260628388467285], [1.2261563109112825, 1.226178268555784, 1.226438108822036, 1.2262525457851228], [1.2261803108565312, 1.2260927394860512, 1.226450592289244, 1.2264153856102011], [1.226291611435127, 1.226285772886538, 1.2261933891132757, 1.2263277657774587], [1.2265462020376658, 1.2261670884749964, 1.2261863431218958, 1.22616803203863], [1.2262980763491, 1.2261198239047466, 1.2262697399617133, 1.226192137964294], [1.2263737766451555, 1.22621173531538, 1.2261128036889453, 1.2263426424465225], [1.2262456086959783, 1.2263299377407524, 1.2262049898730565, 1.2263707445770786], [1.2263224389191656, 1.2262890795372063, 1.2263118883802246, 1.2261472960893864], [1.2261917173224683, 1.2262431858925784, 1.22615749432951, 1.2262334021327435], [1.226266397536502, 1.225999076203741, 1.2262203673630483, 1.2262651822060324], [1.2263182307502007, 1.2261882959593438, 1.226218852284281, 1.226301213862534], [1.226315865091167, 1.2263241340136173, 1.2262851122684162, 1.2262403449949235], [1.2263857924063866, 1.226355559988014, 1.2263458863810701, 1.2262021288456832], [1.2261982478433475, 1.2263965443200446, 1.22625980009216, 1.2261301204696498], [1.2260923590718307, 1.226190055841726, 1.2262611148856943, 1.2263399286139898], [1.2263165154560058, 1.2261779766574525, 1.2262101769016145, 1.2261720724582328], [1.2260641104335912, 1.2263440728817185, 1.2262310135380938, 1.2262133222536866], [1.2264012705031513, 1.226327171203943, 1.226358787134805, 1.2262810306390515], [1.2263007985968275, 1.2263246834006416, 1.226128656861246, 1.226388050685715], [1.2260341612108936, 1.2263117017242413, 1.2260877986782814, 1.2261597487649705], [1.2261632537578846, 1.2262155858650017, 1.226187767393848, 1.2263631905947134], [1.2261267637018434, 1.2261721027561086, 1.2263352031352783, 1.2264948301908563], [1.2262662660365087, 1.2263029213133299, 1.2262900368336123, 1.2262383883863512], [1.2264377848577046, 1.226242675966915, 1.2262640677270096, 1.2261124809848598], [1.2261462320701058, 1.2263386826604192, 1.2262010349266659, 1.2262307792819525], [1.2262392846345325, 1.2261187227275543, 1.2261036139183297, 1.2263754334127517], [1.226185471340196, 1.2261423485831133, 1.2262324386384678, 1.226123676133662], [1.2261107345263473, 1.226093243294457, 1.226341033895037, 1.2263887511526237], [1.2261718723537856, 1.2263001557808955, 1.2264602047742552, 1.22617302720935], [1.2263210470879877, 1.2261625520551842, 1.226240370317954, 1.2262011858271449], [1.226157298650293, 1.2263886167776175, 1.2262414847710528, 1.2263509651233901], [1.2262769717363529, 1.2262628493147292, 1.22631336190624, 1.2262509926515701], [1.2260991966778743, 1.226330590336826, 1.2260952699871968, 1.2261446233300861], [1.2261113399370325, 1.2263211078146785, 1.22628315164988, 1.2263400060669438], [1.2260745940393063, 1.2262144462087214, 1.2261689001688882, 1.2262342003600983], [1.2261694352327568, 1.226244753385697, 1.226281871179992, 1.2261048919324906], [1.2260701210784655, 1.2261913211061342, 1.2262248511523586, 1.2262361082403548], [1.226338511518123, 1.226361128639423, 1.22629175079591, 1.2261892337442677], [1.2263124635286091, 1.2261109161064057, 1.2261781476515683, 1.22629679783725], [1.2262226276535673, 1.226122360689387, 1.2262938646488497, 1.2262247775920603], [1.2262054233484814, 1.2263020136741536, 1.2262552353634644, 1.2263275892538579], [1.2262617183262792, 1.226280635940661, 1.2260931159714339, 1.2263497365531897], [1.226004701657516, 1.2261484411089825, 1.2261119878387123, 1.2262166285018745], [1.2261636888641478, 1.2261220895287936, 1.2260182901076464, 1.2262044298563084], [1.2263295329729145, 1.2262417195545658, 1.226219933397159, 1.226298728831943], [1.226030868432975, 1.2261323224848615, 1.226167375912846, 1.2262628117098373], [1.2262376719332013, 1.2264112674499168, 1.2261985323051672, 1.2262539370750984], [1.2260904608145928, 1.2262788432865075, 1.226243725406852, 1.2262056777745582], [1.2263902368492612, 1.226173837918529, 1.2262876581714406, 1.226363323110716], [1.226211433135056, 1.226350864307278, 1.2262943982849475, 1.2262891484622744], [1.22637133477322, 1.2262991065750095, 1.2262648821600615, 1.2263801231700107], [1.2262808574400639, 1.2262214002420053, 1.2262866555269976, 1.2262128113630733], [1.2262192242214238, 1.226236561368525, 1.226256136541366, 1.2264609279456375], [1.2260455096423024, 1.226316922425255, 1.2260597056369715, 1.2262956700589511], [1.2263597052357136, 1.226150681512715, 1.2262434013953871, 1.2262308979286536], [1.2262482210528745, 1.2262830529023727, 1.2261373785473277, 1.22640715286507], [1.2260053254066747, 1.2261031266413447, 1.2262072405742979, 1.2262211432429333], [1.226413673720838, 1.2262888827449165, 1.2264848238080257, 1.226162068392027], [1.2263604388807918, 1.2261817138235211, 1.2262081322031708, 1.226176720783005], [1.226426692506085, 1.2262626291568504, 1.2262980782489434, 1.2261743424045919], [1.2262312961153812, 1.2263231722970203, 1.226290117054493, 1.226500154028713], [1.2263023855003297, 1.226149041713903, 1.2263424516574217, 1.2262763254904312], [1.2263289205598225, 1.2262631993749464, 1.2263125139628228, 1.226351659164492], [1.2261929685908672, 1.2263747129235991, 1.2264690449649502, 1.226379917775248], [1.2262203593525087, 1.226272287111621, 1.2262646434795386, 1.2262126406949705], [1.2262889238560148, 1.2262013087866648, 1.2261965979454739, 1.2263601523568914], [1.2261410119322413, 1.2264866858971455, 1.2263461750727689, 1.2261131421344993], [1.2263478922556181, 1.226325402372277, 1.2263955001575804, 1.226168826608319], [1.2264138305752084, 1.2261563367264683, 1.2262251019650006, 1.2264175015979326], [1.2261669454975224, 1.2261632062510572, 1.2263305267512976, 1.2262464589149862], [1.2262016538373817, 1.2262576209652105, 1.226066483132118, 1.2262681234532093], [1.2263160919283582, 1.226285761333807, 1.2262312667911806, 1.2261770121156264], [1.22636104816409, 1.2263106874096426, 1.2261608345792618, 1.2262412826163844], [1.2263717809536758, 1.2261075059592204, 1.22635406658978, 1.226283160377019], [1.2264231090391648, 1.2261612203106806, 1.2263201314832333, 1.2262039160916784], [1.226186933053878, 1.2263377923024028, 1.2263309134457656, 1.2264089937401672], [1.22620505168619, 1.22613629481081, 1.226121203565605, 1.2264194088256606], [1.2262403767002432, 1.2262535508266423, 1.2261699644367634, 1.226329155422884], [1.226277364568754, 1.2261851308315506, 1.2262723663941595, 1.2262488884754486], [1.226222646231156, 1.2261581186110733, 1.226224260178429, 1.2262925517934289], [1.2261641835703603, 1.2262506302404363, 1.2260774717947875, 1.2263195475241269], [1.2262474192935788, 1.2262311664240173, 1.2262773124732784, 1.2263314106229988], [1.2260804246155583, 1.2262904768206453, 1.2263233547224426, 1.2263127200174444], [1.226381795941828, 1.2262935310359466, 1.2262561577350672, 1.2261832300352544], [1.2263558331672062, 1.2263292393159546, 1.226431363768884, 1.2262191328236616], [1.2262210101837205, 1.2264093883066152, 1.2261430677285898, 1.226312423515033], [1.2262878001474415, 1.2262084892776488, 1.2262556396669135, 1.2263208450638163], [1.2262246083399215, 1.2264126095447723, 1.2263085778059049, 1.2263548865167073], [1.2262307865320392, 1.2261037512887927, 1.226497800603868, 1.2263496186366516], [1.2261973656897336, 1.2261946279086855, 1.2262145658763943, 1.226323754466328], [1.22636615474651, 1.2263994591930882, 1.2262611954332898, 1.2263122852067734], [1.2262390699206163, 1.2261663491997168, 1.2263894759858283, 1.226383671710448], [1.2263257962435195, 1.226363733168265, 1.2261478007461766, 1.2261953435062753], [1.2261211932493805, 1.226102393371168, 1.2262176409166927, 1.2265156223194977]])
    
    Xn = copy.deepcopy(X)
    Un = copy.deepcopy(U)

    params['X_hist'].append(Xn)
    params['U_hist'].append(Un)

    return params, Xn, Un
