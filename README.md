# BINS
Первые инерциальные системы создавались на базе гиростабилизированной платформы, они делились на три типа — геометрические, аналитические и полуаналитические. Альтернативой платформенной ИНС выступает бессплатформенная инерциальная навигационная система (БИНС). В БИНС акселерометры и гироскопы жестко связаны с корпусом прибора. Функции платформы моделируются математически вычислительной системой. Бесплатформенные системы выгодно отличаются меньшим весом и габаритами, а также возможностью работать при значительных перегрузках.

### Описание алгоритма
В файле функций заданы константы, такие как частота алгоритма (F), период алгоритма (T), ускорение свободного падения (G), скорость вращения земли (U), экваториальный радиус Земли (А), полярный радиус Земли (B), эксцентриситет эллипса (E) и средний радиус Земли (R).<br>

Далее представлена функция block_4, на вход данной функции подается матрица ускорений в связанной системе координат и матрица направляющих косинусов. С помощью библиотеки numpy матрица направляющих косинусов умножается на матрицу ускорений. На выходе функции выдается матрица ускорений в системе географического трехгранника.<br>

В функции block_5 на вход подается матрица ускорений в системе географического трехгранника, матрица скоростей в системе географического трехгранника, матрица угловых скоростей в системе географического трехгранника, высота и широта. Далее вычисляется радиусы, учитывающие несферичность Земли с учетом текущей широты, вычисляются линейные скорости, интегрируя ускорения, и вычисляются угловые скорости, используя вычисленные радиусы и линейные скорости. На выходе функции выдается матрица линейных скоростей в системе географического трехгранника, матриц угловых скоростей в системе географического трехгранника и радиусы, учитывающие несферичность Земли с учетом текущей широты.<br>

В функции block_6 на вход подаются матрицы угловых скоростей связанной системы координат и системы географического трехгранника, а также матрица направляющих косинусов. В данной функции реализуется уравнение Пуассона. На выходе получается матрица направляющих косинусов.<br>

В функции block_7 на вход подается матрица направляющих косинусов. В block_7 реализуется нормализация матрицы, по четным тактам по строкам, а по нечетным по столбцам. На выходе функции выдается матрица направляющих косинусов.<br>

В функции block_8 на вход подается матрица направляющих косинусов. Внутри функции рассчитываются углы ориентации (крен, тангаж и курс), используя элементы матрицы направляющих косинусов. Также написаны граничные условия, защищающие от неопределенности решения. На выходе функции выдается углы ориентации (крен, тангаж и курс).<br>

В функции block_9 на вход подаются линейные скорости в системе географического трехгранника, географические координаты (широта и долгота), высота и радиусы, учитывающие несферичность Земли с учетом текущей широты. В block_9 производится расчет географических координат. На выходе функции выдаются географические координаты.<br>

В функции BL_mat на вход подаются углы ориентации (тангаж, крен и курс). Внутри функции происходит расчет коэффициентов матрицы направляющих косинусов, это необходимо для получения начальной матрицы. На выходе функции получается матрица направляющих косинусов.<br>

В функции transMat реализована операция транспонирования матрицы три на три. На вход подается матрица направляющих косинусов и на выходе получается транспонированная матрица направляющих косинусов.<br>

### Результат работы алгоритма
![image](https://github.com/promodern4/BINS/master/Plane_TU154.gif)
