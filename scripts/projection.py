import math

import numpy as np

from . import ProjectiveGeometry as pg


def create_default_projection_matrix(
    rao_lao_ang=0,
    cran_caud_ang=0,
    img_rot_ang=0,
    pixel_spacing=0.308,
    offset_u=512,
    offset_v=512,
    sid=1200,
    sisod=750,
):
    # P_A = K * R_z_180 * R_x_90 * T * R_s(cran) * R_p(rao))
    rao = math.radians(rao_lao_ang)
    cran = math.radians(cran_caud_ang)
    img_rot = math.radians(img_rot_ang)
    K = get_K(sid, pixel_spacing, offset_u, offset_v)
    R_z_180 = get_Pinhole_correction()
    R_x_90 = get_rx90()
    T = get_translation(sisod)
    R = get_rotation(rao, cran, imgRot=img_rot)

    R_horizontal = np.matrix(np.zeros(shape=(4, 4)))
    R_horizontal[0, 0] = 1
    R_horizontal[1, 1] = -1
    R_horizontal[2, 2] = 1
    R_horizontal[3, 3] = 1
    ##original
    P = K * R_horizontal * R_z_180 * R_x_90 * T * R
    P = K * R_z_180 * R_x_90 * T * R

    return P


def get_uv_point_in_xyz(point_uv, projMat, sid):
    p_i = np.linalg.pinv(projMat)
    point_bp_p = point_uv.backproject(p_i)
    line = point_bp_p.join(get_source_position(projMat))
    det_plane = p3(projMat).get_plane_at_distance(sid)
    return line.meet(det_plane)


def get_K(sid, pixel_spacing, offset_u, offset_v):
    d = sid
    p = pixel_spacing
    ou = offset_u
    ov = offset_v
    K = np.matrix(np.zeros(shape=(3, 4)))
    K[0, 0] = d / p
    K[1, 1] = d / p
    K[0, 2] = ou
    K[1, 2] = ov
    K[2, 2] = 1
    return K


def p1(p):
    return pg.plane_p3(p[0, :])


def p2(p):
    return pg.plane_p3(p[1, :])


def p3(p):
    return pg.plane_p3(p[2, :])


def sp(p):
    return get_source_position(p)


def get_source_position(p):
    return (p1(p).meet(p2(p))).meet(p3(p))


def get_Pinhole_correction():
    R = np.matrix(np.zeros(shape=(4, 4)), dtype=np.float64)
    R[0, 0] = -1
    R[1, 1] = -1
    R[2, 2] = 1
    R[3, 3] = 1
    return R


def get_rx90():
    R = np.matrix(np.zeros(shape=(4, 4)), dtype=np.float64)
    R[0, 0] = 1
    R[1, 2] = 1
    R[2, 1] = -1
    R[3, 3] = 1
    return R


def get_horizontal_flip():
    R_horizontal = np.matrix(np.zeros(shape=(4, 4)), dtype=np.float64)
    R_horizontal[0, 0] = 1
    R_horizontal[1, 1] = -1
    R_horizontal[2, 2] = 1
    R_horizontal[3, 3] = 1
    return R_horizontal


def get_translation(sisod):
    d = sisod
    T = np.matrix(np.zeros(shape=(4, 4)), dtype=np.float64)
    T[0, 0] = 1
    T[1, 1] = 1
    T[2, 2] = 1
    T[3, 3] = 1
    T[1, 3] = -d
    return T


def get_transformation_matrix(alpha=0, beta=0, imgRot=0, tx=0, ty=0, tz=0):
    def s(x):
        return math.sin(x)

    def c(x):
        return math.cos(x)

    ##Rz
    rP = np.matrix(np.zeros(shape=(4, 4)), dtype=np.float64)
    rP[0, 0] = c(alpha)
    rP[0, 1] = s(alpha)
    rP[1, 0] = -s(alpha)
    rP[1, 1] = c(alpha)
    rP[2, 2] = 1.0
    rP[3, 3] = 1.0

    ##Rx
    rS = np.matrix(np.zeros(shape=(4, 4)), dtype=np.float64)
    rS[0, 0] = 1.0
    rS[1, 1] = c(-beta)
    rS[1, 2] = s(-beta)
    rS[2, 1] = -s(-beta)
    rS[2, 2] = c(-beta)
    rS[3, 3] = 1.0

    ##Ry
    rT = np.matrix(np.zeros(shape=(4, 4)), dtype=np.float64)
    rT[1, 1] = 1.0
    rT[0, 0] = c(imgRot)
    rT[0, 2] = s(imgRot)
    rT[2, 0] = -s(imgRot)
    rT[2, 2] = c(imgRot)
    rT[3, 3] = 1.0
    T = rT * rS * rP
    T[0, 3] = tx
    T[1, 3] = ty
    T[2, 3] = tz

    return T


def get_rotation(alpha, beta, imgRot=0):
    def s(x):
        return math.sin(x)

    def c(x):
        return math.cos(x)

    ##Rz
    rP = np.matrix(np.zeros(shape=(4, 4)), dtype=np.float64)
    rP[0, 0] = c(alpha)
    rP[0, 1] = s(alpha)
    rP[1, 0] = -s(alpha)
    rP[1, 1] = c(alpha)
    rP[2, 2] = 1.0
    rP[3, 3] = 1.0

    ##Rx
    rS = np.matrix(np.zeros(shape=(4, 4)), dtype=np.float64)
    rS[0, 0] = 1.0
    rS[1, 1] = c(-beta)
    rS[1, 2] = s(-beta)
    rS[2, 1] = -s(-beta)
    rS[2, 2] = c(-beta)
    rS[3, 3] = 1.0

    ##Ry
    rT = np.matrix(np.zeros(shape=(4, 4)), dtype=np.float64)
    rT[1, 1] = 1.0
    rT[0, 0] = c(imgRot)
    rT[0, 2] = s(imgRot)
    rT[2, 0] = -s(imgRot)
    rT[2, 2] = c(imgRot)
    rT[3, 3] = 1.0
    return rT * rS * rP  ###Ry * Rx * Rz


###get rotation, as implemented in CONRAD
def get_rotation_symmetry(alpha, beta):
    def s(x):
        return math.sin(x)

    def c(x):
        return math.cos(x)

    ##Rz --> implemented as in conrad
    rZ = np.matrix(np.zeros(shape=(4, 4)), dtype=np.float64)
    rZ[0, 0] = c(alpha)
    rZ[0, 1] = -s(alpha)
    rZ[1, 0] = s(alpha)
    rZ[1, 1] = c(alpha)
    rZ[2, 2] = 1.0
    rZ[3, 3] = 1.0

    ##Rx --> implemented as in conrad
    rX = np.matrix(np.zeros(shape=(4, 4)), dtype=np.float64)
    rX[0, 0] = 1.0
    rX[1, 1] = c(beta)
    rX[1, 2] = -s(beta)
    rX[2, 1] = s(beta)
    rX[2, 2] = c(beta)
    rX[3, 3] = 1.0
    return rZ * rX


def rodriguez(axis: np.matrix, angle_radians, make_matrix_homogen=False):
    return get_rotation_matrix_by_axis_and_angle(
        axis, math.degrees(angle_radians), make_matrix_homogen=make_matrix_homogen
    )


def get_rotation_matrix_by_axis_and_angle(axis, angle_deg, make_matrix_homogen=False):
    """
    This method is kind of depricated...we do not want to give angles as function input. We always expect radians.
    However, this function takes angles...
    Use rodriguez instead
    :param axis:
    :param angle_deg:
    :param make_matrix_homogen:
    :return:
    """
    angle_rad = angle_deg * math.pi / 180
    cos_theta = math.cos(angle_rad)
    sin_theta = math.sin(angle_rad)
    axis = axis / np.linalg.norm(axis)

    ux = axis[0, 0]
    uy = axis[1, 0]
    uz = axis[2, 0]
    u_skew = np.matrix([[0, -uz, uy], [uz, 0, -ux], [-uy, ux, 0]])
    u_tensor = np.matrix(
        [[ux * ux, ux * uy, ux * uz], [ux * uy, uy * uy, uy * uz], [ux * uz, uy * uz, uz * uz]]
    )
    id = np.matrix(np.diag(np.ones(3)))
    r = cos_theta * id + sin_theta * u_skew + (1 - cos_theta) * u_tensor
    if make_matrix_homogen:
        R = np.matrix(np.zeros(shape=(4, 4)))
        R[0, 0:3] = r[0, :]
        R[1, 0:3] = r[1, :]
        R[2, 0:3] = r[2, :]
        R[3, 3] = 1
        return R
    else:
        return r


def rx(radians, homo=False):
    R = np.matrix(np.diag([1.0, 1.0, 1.0, 1.0]))
    R[1, 1] = math.cos(radians)
    R[2, 2] = math.cos(radians)
    R[1, 2] = -math.sin(radians)
    R[2, 1] = math.sin(radians)
    if homo:
        return R
    else:
        return R[0:3, 0:3]


def ry(radians, homo=False):
    R = np.matrix(np.diag([1.0, 1.0, 1.0, 1.0]))
    R[0, 0] = math.cos(radians)
    R[2, 2] = math.cos(radians)
    R[0, 2] = math.sin(radians)
    R[2, 0] = -math.sin(radians)
    if homo:
        return R
    else:
        return R[0:3, 0:3]


def rz(radians, homo=False):
    R = np.matrix(np.diag([1.0, 1.0, 1.0, 1.0]))
    R[0, 0] = math.cos(radians)
    R[1, 1] = math.cos(radians)
    R[0, 1] = -math.sin(radians)
    R[1, 0] = math.sin(radians)
    if homo:
        return R
    else:
        return R[0:3, 0:3]


def get_alpha_beta_from_normal(nx, ny, nz, rgt=0):
    """
    Method to get primary (rao/lao), secondary (cran/caud) and tertiary (img_rot) of projection
    matrix, as presented in the thesis.
    :param proj_mat:
    :param rgt: optinalPrameter for debugging
    :return:
    """
    ###probably this H_xy stuff is not necessary however we keep it cause there is no time to further test stuff...
    H_xy = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float64)
    e_x = np.matrix([1, 0, 0], dtype=np.float64).T
    e_z = np.matrix([0, 0, 1], dtype=np.float64).T
    e_y = np.matrix([0, 1, 0], dtype=np.float64).T
    m_3 = np.matrix([nx, ny, nz], dtype=np.float64).T

    ###alpha
    sign_alpha = np.sign((e_x.T * m_3)[0, 0])
    if sign_alpha == 0:
        sign_alpha = 1

    alpha_abs = angulation(-e_y, H_xy * m_3)
    alpha = alpha_abs * sign_alpha

    ####beta ###not sure about sign of beta...need to verify this
    ##...the sign is right, but make sure to map it in the correct anatomical angle
    sign_beta = np.sign((-e_z.T * m_3)[0, 0])
    if sign_beta == 0:
        sign_beta = 1
    beta_abs = angulation(-e_y, rz(alpha).T * m_3)
    # beta_abs = angulation(Rz(alpha) * -e_y, m_3)
    beta = sign_beta * beta_abs

    return (math.degrees(alpha), math.degrees(beta))


def get_alpha_beta_gamma_from_p(proj_mat, rgt=0):
    """
    Method to get primary (rao/lao), secondary (cran/caud) and tertiary (img_rot) of projection
    matrix, as presented in the thesis.
    :param proj_mat:
    :param rgt: optinalPrameter for debugging
    :return:
    """
    H_xy = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float64)
    e_x = np.matrix([1, 0, 0], dtype=np.float64).T
    e_z = np.matrix([0, 0, 1], dtype=np.float64).T
    e_y = np.matrix([0, 1, 0], dtype=np.float64).T
    m_3 = proj_mat[2, 0:3].T / np.linalg.norm(proj_mat[2, 0:3])

    p1 = pg.plane_p3(proj_mat[0, :])
    p3 = pg.plane_p3(proj_mat[2, :])
    v = np.matrix(p1.meet(p3).dir()).T / np.linalg.norm(p1.meet(p3).dir())

    ###alpha
    sign_alpha = np.sign((e_x.T * m_3)[0, 0])
    if sign_alpha == 0:
        sign_alpha = 1

    alpha_abs = angulation(-e_y, H_xy * m_3)
    alpha = alpha_abs * sign_alpha

    ####beta ###not sure about sign of beta...need to verify this
    ##...the sign is right, but make sure to map it in the correct anatomical angle
    sign_beta = np.sign((-e_z.T * m_3)[0, 0])
    if sign_beta == 0:
        sign_beta = 1
    beta_abs = angulation(-e_y, rz(alpha).T * m_3)
    # beta_abs = angulation(Rz(alpha) * -e_y, m_3)
    beta = sign_beta * beta_abs

    ####gamma
    gamma_abs = angulation(-e_z, rx(beta).T * v)
    sign_gamma = np.sign(((rx(beta).T * v).T * -e_y)[0, 0])
    if sign_gamma == 0:
        sign_gamma = 1
    gamma = gamma_abs * sign_gamma

    return (math.degrees(alpha), math.degrees(beta), math.degrees(gamma))


def angulation(a, b):
    """
    :param a: Vector a
    :param b: Vector b
    :return: the angle in radians between vector a and b
    """
    if np.linalg.norm(a) == math.acos(0):
        return 0
    if np.linalg.norm(b) == math.acos(0):
        return 0
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return math.acos(min((a.T * b)[0, 0], 1))


def create_p_aligned(P, lp, P_i=None):
    """
    :param ProjMatirx:
    :param lp:
    :return: Projection matrix that has its central ray aligned, i.e. parallel, to the backprojection plane of lp
    """
    if P_i is None:
        P_i = np.linalg.pinv(P)

    # Source position of P
    sp = get_source_position(P)

    # Principal ray of P (I think)
    m = P[2, 0:3]

    ###get backprojection
    l_bp = lp.backproject(P_i)  # Line corresponding to P1_inv @ sk(lp) @ P1_inv.T
    ###and the direction of this...make it safer maybe...
    # l_bpd is the same as r.
    l_bpd = np.matrix(np.matrix(l_bp.dir()).T)

    ### calculate the normal plane cp. So this is a vector orthogonal to the principle ray and
    ### the line in 3D. l_bpd is really the same as r.
    n_cp = np.cross(l_bpd.T, m).T

    ### calculate the normal of the backprojection plane
    l_lsp = np.matrix(l_bp.get_point_on_line().join(sp).dir()).T
    n_bpp = np.matrix(np.cross(l_bpd.T, l_lsp.T)).T
    # calculate angle...normailzation is done by angulation fkt

    # Okay, so this is the angle between plane cp and plane bpp
    phi = angulation(n_cp, n_bpp)
    # Dependent on which side (w.r.t central plane) we are, we need to rotate in + or - direction
    if (m * n_bpp)[0, 0] < 0:
        phi = -phi
    R = rodriguez(np.matrix(l_bp.dir()).T, phi, True)
    P_a = P * R
    return P_a


def get_table_movement(P_v1, P_v1algn):
    sp = get_source_position(P_v1)
    sp_a = get_source_position(P_v1algn)
    t = np.array(sp_a.e()) - np.array(sp.e())
    H_t = np.matrix(np.diag([1, 1, 1, 1]), dtype=np.float64)
    H_t[0:3, 3] = np.matrix(t).T
    # H_t[0, 3] = 1000
    return H_t


def calculate_p_v2(p_v1, xi, r):
    """
    :param p_v1: First View
    :param xi: amount of radians that we want to rotate
    :param r: axis around we want to rotate
    :return: p_v2 Second View
    """
    RotMat = rodriguez(r, xi, make_matrix_homogen=True)
    p_v2 = p_v1 * RotMat
    return p_v2


def calculate_rotation_axis(P, lp, P_i=None, chooseSaveMethod=True):
    if P_i is None:
        P_i = np.linalg.pinv(P)
    if chooseSaveMethod:
        r = (lp.backproject(P_i).join(sp(P))).meet(p3(P))
    else:
        r = lp.backproject(P_i)
    return np.matrix(r.dir()).T


def detector_meet_points(ul_detector, ur_detector, ll_detector, lr_detector, line):
    ul = ul_detector.e()
    ur = ur_detector.e()
    ll = ll_detector.e()
    lr = lr_detector.e()

    left_right_tmin = ul[1]
    left_right_tmax = ll[1]
    up_down_min = ul[0]
    up_down_max = ur[0]

    left = ul_detector.join(ll_detector)
    right = ur_detector.join(lr_detector)
    down = ll_detector.join(lr_detector)
    up = ul_detector.join(ur_detector)

    meet_points = []
    left_meet = line.meet(left)
    lme = left_meet.get_euclidean_point()
    if lme[1] > left_right_tmin and lme[1] < left_right_tmax:
        meet_points.append(lme)
    right_meet = line.meet(right)
    rme = right_meet.get_euclidean_point()
    if rme[1] > left_right_tmin and rme[1] < left_right_tmax:
        meet_points.append(rme)

    down_meet = line.meet(down)
    dme = down_meet.get_euclidean_point()
    if dme[0] > up_down_min and dme[0] < up_down_max:
        meet_points.append(dme)

    up_meet = line.meet(up)
    ume = up_meet.get_euclidean_point()
    if ume[0] > up_down_min and ume[0] < up_down_max:
        meet_points.append(ume)

    return meet_points
