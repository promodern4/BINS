import numpy as np
from funcs import (block_4,
                   block_5,
                   block_6,
                   block_7,
                   block_8,
                   block_9,
                   BL_mat
                   )


def main():
    with open('NavData_FOG_Car.txt', 'r') as file:
        lines = file.readlines()
    file.close()

    gyro_row_num = 16
    height_row_num = 29
    matrix_of_guiding_cos = BL_mat(1.763, 1.121, 91.661)
    fi = 55.6023783315
    la = 38.0885100000


# ---------------------------------------------------------------- Стоим, не двигаемся! (к чему будем добавлять первый шаг)
    vels_LL = [0, 0, 0]
    a_vels_LL = [0, 4.182 * 10 ** -5, 5.973 * 10 ** -5]


# ---------------------------------------------------------------- Открылваем файл для записи
    with open('resultL.txt', 'a') as file:

# ---------------------------------------------------------------- Начинаем основной цикл алгоритма      
        for index in range(60000, 600000):


# ---------------------------------------------------------------- Вычленяем из исходных данных показания датчиков и значение высоты
            accs_B = []
            a_vels_B = []

            line = lines[index].split()

            for i in range(gyro_row_num, gyro_row_num + 3):
                a_vels_B.append(float(line[i]))
                accs_B.append(float(line[i+3]))
            height = float(line[height_row_num])
# ---------------------------------------------------------------- Пересчитываем ускорения из Body в LocalLevel
            accs_LL = block_4(accs_B, matrix_of_guiding_cos)
# ---------------------------------------------------------------- Пересчитываем скорости в LL и радиусы Земли (a_vels_LL - рад/с)
            vels_LL, a_vels_LL, r_fi, r_la = block_5(accs_LL, vels_LL, a_vels_LL, height, fi)
# ---------------------------------------------------------------- Решаем уравнение Пуссона
            matrix_of_guiding_cos = block_6(a_vels_B, a_vels_LL, matrix_of_guiding_cos)
# ---------------------------------------------------------------- Нормализуем матрицу
            matrix_of_guiding_cos = block_7(matrix_of_guiding_cos, index % 2)
# ---------------------------------------------------------------- Расчет углов
            theta, gamma, psi = block_8(matrix_of_guiding_cos)
# ---------------------------------------------------------------- Расчет координат
            fi, la = block_9(vels_LL[0], vels_LL[1], fi, la, height, r_fi, r_la)
# ---------------------------------------------------------------- Результат на запись в файл
            result = [fi, la, gamma, theta, psi, vels_LL[0], vels_LL[1]]
            file.write("{} {} {} {} {} {} {}\n".format(*result))
# ----------------------------------------------------------------
# ----------------------------------------------------------------

    file.close()


if __name__ == '__main__':
    main()