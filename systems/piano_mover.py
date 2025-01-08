import numpy as np
from proximity.proximity import proximity_mrp
from proximity.proximity_gradient import proximity_gradient
from primitives.misc_primitive_constructor import create_rect_prism


def dynamics(params, x, u, k):
    """
    Computes the dynamics of the system.

    Args:
        params (dict): Simulation parameters.
        x (ndarray): State vector.
        u (ndarray): Control input vector.
        k (int): Current timestep index.

    Returns:
        ndarray: Updated state derivative.
    """
    r = x[:2]
    v = x[2:4]
    theta = x[4]
    omega = x[5]

    return np.concatenate([v, u[:2], [omega], [u[2] / 100]])


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

def inequality_constraints_x(params, x):
    """
    Calculates state-based inequality constraints.

    Args:
        params (dict): Simulation parameters.
        x (ndarray): Current state.

    Returns:
        ndarray: Constraint violations for the current state.
    """
    params['P_vic'].r = np.array([x[0], x[1], 0])
    params['P_vic'].p = np.array([0, 0, 1]) * np.tan(x[4] / 4)
    
    contacts = []
    for obs in params['P_obs']:
        alpha, _ = proximity_mrp(params['P_vic'], obs, verbose=False)
        contacts.append(1 - alpha)
        
    return np.array(contacts)

def inequality_constraints_x_grad(params, x):
    """
    Calculates the gradient of the state-based inequality constraints.

    Args:
        params (dict): Simulation parameters.
        x (ndarray): Current state.

    Returns:
        ndarray: Gradient of the constraint violations.
    """

    rx, ry, vx, vy, theta, omega = x
    dp_dtheta = np.array([0, 0, 1]) * (1 / (4 * np.cos(theta / 4) ** 2))
    params["P_vic"].r = np.array([*x[:2], 0])
    params["P_vic"].p = np.array([0, 0, 1]) * np.tan(x[4] / 4)

    Js = []
    for obs in params["P_obs"]:
        alpha, d_alpha_d_state  = proximity_gradient(params["P_vic"], obs)
        Js.append(d_alpha_d_state)
        
    contact_J = []
    for i in range(len(params["P_obs"])):
        c = np.concatenate([-Js[i][:2].T, [0, 0], np.array([-np.dot(Js[i][3:6].T, dp_dtheta)]), [0]])
        contact_J.append(c)

    return np.array(contact_J)

def inequality_constraints_u(params, u):
    """
    Calculates input-based inequality constraints.

    Args:    
        params (dict): Simulation parameters.
        u (ndarray): Control input vector.

    Returns:
        ndarray: Constraint violations for the input.
    """
    u_max = params['u_max']
    u_min = params['u_min']
    return np.concatenate([u - u_max, -u + u_min])

def inequality_constraints_u_grad(params, u):
    """
    Calculates the gradient of the input-based inequality constraints.

    Args:
        params (dict): Simulation parameters.
        u (ndarray): Control input vector.

    Returns:
        ndarray: Gradient of the constraint violations.
    """
    
    nu = params["nu"]
    return np.vstack([np.eye(nu), -np.eye(nu)])


def initialize_piano_mover():
    """
    Initializes the parameters for the piano mover system.

    """

    print("Initializing parameters for the piano mover...")
    nx = 6
    nu = 3
    N = 80
    dt = 0.1
    max_linesearch_iters = 20
    atol = 4e-2
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

    # states are [rx, ry, vx, vy, theta, omega]
    x0 = np.array([1.5, 1.5, 0, 0, 0, 0])
    xg = np.array([3.5, 3.7, 0, 0, np.deg2rad(90), 0])

    Xref = [np.copy(xg) for _ in range(N)]
    Uref = [np.zeros(nu) for _ in range(N - 1)]
    
    # These are the cost weight matrices
    Q = np.diag(np.ones(nx)) # penalizes state deviations from the reference
    Qf = np.diag(np.ones(nx)) # penalizes final state deviations from the reference
    R = np.diag([1, 1, 0.001]) # penalizes the magnitude of the control input

    P_vic = create_rect_prism(2.5, 0.15, 0.01)

    P_obs = [
        create_rect_prism(3.0, 3.0, 1.0),
        create_rect_prism(4.0, 1.0, 1.0),
        create_rect_prism(1.0, 5.0, 1.1)
    ]

    P_obs[0].r = [1.5, 3.5, 0.0]
    P_obs[1].r = [2, 0.5, 0]
    P_obs[2].r = [4.5, 2.5, 0]


    u_min = -200 * np.ones(nu)
    u_max = 200 * np.ones(nu)

    x_min = -200 * np.ones(nx)
    x_max = 200 * np.ones(nx)

    params = {
        'nx': nx, # number of states
        'nu': nu, # number of control inputs
        'ncx': len(P_obs), # number of state constraints
        'ncu': 2 * nu, # number of control constraints
        'N': N, # number of timesteps
        'Q': Q, # state cost matrix
        'R': R, # control cost matrix
        'Qf': Qf, # final state cost matrix
        'u_min': u_min, # control input lower bound
        'u_max': u_max, # control input upper bound
        'x_min': x_min, # state lower bound
        'x_max': x_max, # state upper bound
        'Xref': Xref, # reference state trajectory
        'Uref': Uref, # reference control trajectory
        'dt': dt, # timestep duration
        'P_obs': P_obs, # list of obstacles
        'P_vic': P_vic, # the robot
        'max_linesearch_iters': max_linesearch_iters, # maximum number of linesearch iterations
        'atol': atol, # absolute tolerance
        'max_iters': max_iters, # maximum number of iterations
        'X_hist': X_hist, # history of state trajectories
        'U_hist': U_hist, # history of control trajectories
        'hx_hist': hx_hist, # history of state constraint violations
        'hu_hist': hu_hist, # history of control constraint violations
        'reg_min': reg_min, # minimum regularization parameter
        'reg': reg, # regularization parameter
        'reg_max': reg_max, # maximum regularization parameter
        'rho': rho, # penalty parameter
        'phi': phi, # used to update the regularization parameter
        'convio_tol': convio_tol, # convergence tolerance
        'system': 'piano_mover'
    }

    print("Initializing state and control trajectories...")
    X = [np.copy(x0) for _ in range(N)]

    # IMPORTANT NOTE: The following values are hardcoded for the initial positions and orientations of the obstacles.
    # In the original Julia code implementation, these values were obtained "randomly" with seeed value 2. 
    # Since the same seed will not produce the same values in Python, we just took the values from the original implementation
    # and put them here to have the same experiment and results
    U = np.array([[-5.737244600260246e-5, 0.017353762138709008, -0.010499249292429236], [0.005753709361353411, -0.0036738997477480557, 0.010071663760294896], [-0.012757079018632909, -0.004260902600860004, 0.006428270594194301], [-0.01399624395864564, -0.013655014696821087, -0.010833803853072635], [0.013889884107295043, 0.0025768432578700535, 0.009495456360044525], [-0.0008335092685759774, 0.015771204395963797, 0.01308596285801138], [-0.009354855316113407, 0.022227791310293638, -0.015076266600792008], [-0.009563452315543272, -0.01485916028223613, 0.0013248192409683762], [-0.0076064584475215484, -0.0019758440222590505, 0.005233180613375706], [-0.0045755094908397406, 0.010948591718759726, 0.01223124750654056], [-0.002940722487079552, 0.0057685591697717855, 0.003627412138006672], [-0.006204034551844231, 0.0007213065417140196, -0.0012552901732286762], [0.010529030460467846, 0.007511358098080348, -0.004190471036995896], [-0.007878193722265461, 0.006824127129899018, 0.019253108008110136], [0.002864285963979535, 0.013167399858186725, 0.013297774834448526], [0.000983039667839325, -0.0013554283986242813, 0.003367230609802889], [-0.0028761815520658113, 0.011699950975920257, 0.009266279942495301], [-0.0018345938757698822, 0.004753331420863974, 0.010269311489307953], [-0.025670512738033197, -0.006755876566922168, 0.0015936838411553414], [0.0016480566266824212, 0.0008346995263954868, -0.02151785397829064], [0.0053442715954029815, -0.01406246501551248, 0.00683678023252568], [-0.006703495154195287, -0.008608476896737206, -0.0023521817905025814], [-0.012407713963690725, 0.012595827527768227, 0.004937230492577874], [-0.0010616156646020998, -0.01584984894727407, 0.004900843109878008], [0.01305267283254087, -0.007282078266205122, 0.011249857431661198], [0.006606408280868398, -0.0019806312301744768, 0.016470739064966367], [0.004802439374738011, 0.013885789040406141, 0.009014055205390219], [0.0052888312538386965, 0.013297646085688309, -0.009217217590850257], [0.013286720026877226, 0.005263614372977995, -0.0008508789851517715], [0.009640116445102674, 0.01478475716714185, 0.013572472551165343], [0.014004109050465506, -0.016459679533685382, 0.006627089565654273], [-0.010534701904340485, 0.011875922357170235, 0.0004855181183309595], [0.00860466391188423, 0.010912868349301292, 0.0022741803795421145], [0.010804789514561984, -0.022617877860439672, -0.005204240859130257], [0.006331477465971246, 0.016960507391164656, -0.001547653970824142], [0.002899109566977512, 0.005942322405282904, -0.009201265737973797], [0.005149005147851648, -0.002895286300656861, 7.063456610411353e-5], [0.011873286488334491, 0.0157644262022363, -0.01121442521298041], [-0.0032521233810789687, 0.008418618464080091, -0.004366476427646501], [-0.022276666730295987, 0.017781986159837445, -0.0017012802908841149], [0.004512106343315697, 0.005379085767942904, 0.002913064469567252], [0.010565860047936668, -0.00681572149425528, 0.0005971836070291437], [-0.013079656536932995, 0.021088250472574243, -0.0020462913127031798], [-0.02810343030433742, 0.011101542118306322, -0.010589278596782374], [0.010870176696707666, 0.02042540643425835, 0.003797882125360135], [0.007058682971627416, 0.013050556144315899, -0.012157978095051813], [0.00899655340436374, 0.006867261153472466, 0.009048172808753694], [0.0059556626557446726, -0.006063069346584732, 0.009624979949539722], [0.001970511753905374, -0.012309639750182575, 0.006144801547429508], [-0.0024374988495411737, 0.0013365597121188027, -0.010071345099069142], [0.019294416113249776, 0.006076261708517241, -0.003670825588302479], [-0.003403591370706819, -0.02702130088403541, -0.005568128325618383], [-0.004969493798967757, 0.010961865007109692, 0.013210309515391714], [-0.002906647766387922, -0.0015452486792222786, 0.002995855475665709], [0.012259658280399382, 0.001531485985776024, 0.009771153396326324], [0.004536257211061856, 0.011497768733009195, -0.008351397466505026], [-0.014213293793717037, 0.0014938614874661564, -0.00943392007273104], [0.020936651658201147, -0.0013573668230998422, 0.0010434794038996042], [-0.0038462630904338374, -0.01259278306446005, -0.0028784228581587162], [-0.0022686191323123698, -0.011583855402897915, 0.010639408870820273], [-0.006574844635000492, -0.0022442101480024393, 0.017280687164755372], [0.004309967183204165, -0.0003933903371953597, -0.003115678125888011], [-0.003973298848921057, 0.008003057997766717, -0.00999439762764241], [0.001335040789967567, -0.014228855853359184, 0.00030736678349147477], [0.006039692807571979, -0.006641634260680348, -0.005044519727261649], [-0.008239293977365753, 0.0020064481353764445, 0.007195824775072087], [0.016665075745221043, -0.0017969458106066936, -0.009879365528517682], [0.00364913686032916, -0.012270973633463014, -0.0013260731893863893], [0.0008956707952600792, -0.013596694462057522, -0.008740273357269256], [-0.014426289354102714, 0.008261209385562733, -0.0006611615693516227], [0.009422452337661522, 0.005516349058197596, -0.00442235966858532], [-0.002091606522008061, -0.010080700884741367, -0.001488747574923605], [0.0030148428672197207, 0.002133724204790347, -0.004746690159216579], [-0.006119716874876675, 0.0003994815554527361, 0.008916916654116308], [-0.0006340572737427659, -0.007385213555848181, -0.005038402780246292], [0.02254435644093704, 0.015206606762722362, 0.01605538900438031], [0.006686763332698393, -0.01840026572553052, 0.0012440740958287748], [0.005386290176175485, 0.011399829723178627, -0.005938148908974193], [0.00846748051818129, 0.019235972030493566, -0.006419025527832839]])

    params['X_hist'].append(X)
    params['U_hist'].append(U)

    return params, X, U