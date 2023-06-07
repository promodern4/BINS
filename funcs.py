import numpy as np
from sklearn.preprocessing import normalize


# ----------------------------------------------------------------
# Переменные

# Частота алгоритма, Гц
F = 100

# Период алгоитма, сек
T = 1/F

# Ускорение свободного падения, м/с^2
G = 9.81523

# Скорость вращени Земли, рад/с
U = 7.292115 * 10 ** -5

# Экваториальный радиус Земли, м
A = 6.378 * 10 ** 6

# Полярный радиус Земли, м
B = 6.357 * 10 ** 6

# Эксцентриситет эллипса
E = (A ** 2 - B ** 2) ** 0.5 / A

# Средний радиус Земли, м
R = 6.371 * 10 ** 6
# ----------------------------------------------------------------


# ---------------------------------------------------------------- Перевод укорений из Body в LocalLevel
def block_4(input_accs: list,
            matrix_of_guiding_cos: list) -> list:
    """Переводит ускорения из системы Body в систему LocalLevel

    >>> block_4([1, 1, 1], [[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    [3, 3, 3]
    """
    reversed_input_accs = [[_] for _ in input_accs]
    result = np.dot(matrix_of_guiding_cos,
                    reversed_input_accs)
    output_accs = [i[0] for i in result]

    return output_accs # [fe, fn, fup]
# ----------------------------------------------------------------


# ---------------------------------------------------------------- Расчет линейных скоростей и угловых скоростей
def block_5(accs: list, vels: list, a_vels: list, h: float, fi: float):
    f_e, f_n, f_up = accs[0], accs[1], accs[2]
    v_e, v_n, v_up = vels[0], vels[1], vels[2]

    v_e_1, v_n_1, v_up_1 = v_e, v_n, v_up

    omega_e, omega_n, omega_up = a_vels[0], a_vels[1], a_vels[2]
    fi = np.pi * fi / 180
    
    r_fi = (R * (1 - E ** 2)) / (1 - E ** 2 * np.sin(fi)) ** (1.5)
    r_lam = (R * (1 - E ** 2)) / (1 - E ** 2 * np.sin(fi)) ** (0.5)

    v_e_1 += (f_e - (omega_n * v_up) + (omega_up * v_n) - (U * v_up * np.cos(fi)) + (U * v_n * np.sin(fi))) * T
    v_n_1 += (f_n + (omega_e * v_up) - (omega_up * v_e) - (U * v_e * np.sin(fi))) * T
    # v_up_1 += (f_up - (omega_e * v_n) + (omega_n * v_e) + (U * v_e * np.cos(fi)) - G) * T
    v_up_1 = 0

    omega_e = - v_n_1 / (r_fi + h)
    omega_n = v_e_1 / (r_lam + h) + U * np.cos(fi)
    omega_up = v_e_1 * round(np.tan(fi), 4) / (r_lam + h) + U * np.sin(fi)

    output_velocities = [v_e_1, v_n_1, v_up_1]
    output_ang_velocities = [omega_e, omega_n, omega_up]

    return output_velocities, output_ang_velocities, r_fi, r_lam
# ----------------------------------------------------------------


# ---------------------------------------------------------------- Уравнение Пуассона
def block_6(a_vels_B, a_vels_LL, CBLL):
    omega_xb = a_vels_B[0]*np.pi/180
    omega_yb = a_vels_B[1]*np.pi/180
    omega_zb = a_vels_B[2]*np.pi/180
    omega_E = a_vels_LL[0]
    omega_N = a_vels_LL[1]
    omega_Up = a_vels_LL[2]


    omegaB = np.array([[0, -omega_zb, omega_yb],
                       [omega_zb, 0, -omega_xb],
                       [-omega_yb, omega_xb, 0]])
    
    omegaENUP = np.array([[0, -omega_Up, omega_N],
                          [omega_Up, 0, -omega_E], 
                          [-omega_N, omega_E, 0]])
    
    CBLL = (CBLL + np.dot(CBLL, omegaB) * T - np.dot(omegaENUP, CBLL)
            * T)
    CBLL = [[CBLL[j][i] for i in range(3)] for j in range(3)]
    return CBLL
# ----------------------------------------------------------------


# ---------------------------------------------------------------- Нормализация матрицы
def block_7(CBLL, norm_pointer):
    if norm_pointer: # по строкам
        CBLL[0][0] = CBLL[0][0] / ((CBLL[0][0]) ** 2 + (CBLL[0][1]) ** 2 + (CBLL[0][2]) ** 2) ** 0.5
        CBLL[0][1] = CBLL[0][1] / ((CBLL[0][0]) ** 2 + (CBLL[0][1]) ** 2 + (CBLL[0][2]) ** 2) ** 0.5
        CBLL[0][2] = CBLL[0][2] / ((CBLL[0][0]) ** 2 + (CBLL[0][1]) ** 2 + (CBLL[0][2]) ** 2) ** 0.5

        CBLL[1][0] = CBLL[1][0] / ((CBLL[1][0]) ** 2 + (CBLL[1][1]) ** 2 + (CBLL[1][2]) ** 2) ** 0.5
        CBLL[1][1] = CBLL[1][1] / ((CBLL[1][0]) ** 2 + (CBLL[1][1]) ** 2 + (CBLL[1][2]) ** 2) ** 0.5
        CBLL[1][2] = CBLL[1][2] / ((CBLL[1][0]) ** 2 + (CBLL[1][1]) ** 2 + (CBLL[1][2]) ** 2) ** 0.5

        CBLL[2][0] = CBLL[2][0] / ((CBLL[2][0]) ** 2 + (CBLL[2][1]) ** 2 + (CBLL[2][2]) ** 2) ** 0.5
        CBLL[2][1] = CBLL[2][1] / ((CBLL[2][0]) ** 2 + (CBLL[2][1]) ** 2 + (CBLL[2][2]) ** 2) ** 0.5
        CBLL[2][2] = CBLL[2][2] / ((CBLL[2][0]) ** 2 + (CBLL[2][1]) ** 2 + (CBLL[2][2]) ** 2) ** 0.5
    else: # по столбцам
        CBLL[0][0] = CBLL[0][0] / ((CBLL[0][0]) ** 2 + (CBLL[1][0]) ** 2 + (CBLL[2][0]) ** 2) ** 0.5
        CBLL[1][0] = CBLL[1][0] / ((CBLL[0][0]) ** 2 + (CBLL[1][0]) ** 2 + (CBLL[2][0]) ** 2) ** 0.5
        CBLL[2][0] = CBLL[2][0] / ((CBLL[0][0]) ** 2 + (CBLL[1][0]) ** 2 + (CBLL[2][0]) ** 2) ** 0.5

        CBLL[0][1] = CBLL[0][1] / ((CBLL[0][1]) ** 2 + (CBLL[1][1]) ** 2 + (CBLL[2][1]) ** 2) ** 0.5
        CBLL[1][1] = CBLL[1][1] / ((CBLL[0][1]) ** 2 + (CBLL[1][1]) ** 2 + (CBLL[2][1]) ** 2) ** 0.5
        CBLL[2][1] = CBLL[2][1] / ((CBLL[0][1]) ** 2 + (CBLL[1][1]) ** 2 + (CBLL[2][1]) ** 2) ** 0.5

        CBLL[0][2] = CBLL[0][2] / ((CBLL[0][2]) ** 2 + (CBLL[1][2]) ** 2 + (CBLL[2][2]) ** 2) ** 0.5
        CBLL[1][2] = CBLL[1][2] / ((CBLL[0][2]) ** 2 + (CBLL[1][2]) ** 2 + (CBLL[2][2]) ** 2) ** 0.5
        CBLL[2][2] = CBLL[2][2] / ((CBLL[0][2]) ** 2 + (CBLL[1][2]) ** 2 + (CBLL[2][2]) ** 2) ** 0.5
    return CBLL
# ----------------------------------------------------------------


# ---------------------------------------------------------------- Расчет углов
def block_8(CBLL):
    C12 = CBLL[0][1]
    C22 = CBLL[1][1]
    C31 = - CBLL[2][0]
    C32 = CBLL[2][1]
    C33 = CBLL[2][2]

    C = (C31 ** 2 + C33 ** 2) ** 0.5
    if C == 0:
        if C32 > 0:
            thetaNew = np.pi / 2
        else:
            thetaNew = - np.pi / 2
    else:
        thetaNew = np.arctan(C32 / C)

    if C33 == 0:
        if C31 > 0:
            gammaNew = np.pi / 2
        else:
            gammaNew = - np.pi / 2
    else:
        if C33 > 0:
            gammaNew = np.arctan(C31 / C33)
        else:
            if C31 < 0:
                gammaNew = - np.pi + np.arctan(C31 / C33)
            else:
                gammaNew = np.pi + np.arctan(C31 / C33)
    
    if C22 == 0:
        if C12 > 0:
            HNew = np.pi / 2
        else:
            HNew = - np.pi / 2
    else:
        if C22 > 0:
            HNew = np.arctan(C12 / C22)
        else:
            if C12 > 0:
                HNew = np.pi + np.arctan(C12 / C22)
            else:
                HNew = np.pi + np.arctan(C12 / C22)
    
    if HNew < 0:
        HNew += 2 * np.pi

    return thetaNew*180/np.pi, gammaNew*180/np.pi, HNew*180/np.pi
# ----------------------------------------------------------------


# ---------------------------------------------------------------- Расчет навигационных параметров
def block_9(v_e, v_n, fi, lamda, h, r_fi, r_lamda):
    lamda += (v_e / ((r_lamda + h) * np.cos(fi*np.pi/180))) * T * 180/np.pi
    fi += (v_n / (r_fi + h)) * T * 180/np.pi
    return fi, lamda
# ----------------------------------------------------------------


# ---------------------------------------------------------------- Начальная матрица направляющих косинусов
def transMat(func):
    def wrapper(*args):
        m = func(*args)
        m_t = [[0 for i in range(3)] for j in range(3)]
        for i in range(3):
            for j in range(3):
                m_t[j][i] = m[i][j]
        return m_t
    return wrapper

@transMat
def BL_mat(t, g, p):
    p = p * np.pi / 180
    t = t * np.pi / 180
    g = g * np.pi / 180
    C11 = np.cos(g) * np.cos(p) + np.sin(t) * np.sin(g) * np.sin(p)
    C12 = - np.cos(g) * np.sin(p) + np.sin(t) * np.sin(g) * np.cos(p)
    C13 = - np.cos(t) * np.sin(g)
    C21 = np.cos(t) * np.sin(p)
    C22 = np.cos(t) * np.cos(p)
    C23 = np.sin(t)
    C31 = np.sin(g) * np.cos(p) - np.sin(t) * np.cos(g) * np.sin(p)
    C32 = - np.sin(g) * np.sin(p) - np.sin(t) * np.cos(g) * np.cos(p)
    C33 = np.cos(t) * np.cos(g)
    CBLL = [[C11, C12, C13], [C21, C22, C23], [C31, C32, C33]]
    return CBLL
# ----------------------------------------------------------------