# Домашнее задание № 2
## Метод наименьших квадратов

В файле [ccd_v2.fits](https://disk.yandex.ru/d/PQ4zZ-SkA5kzBw) приведены данные эксперимента с приёмником изображений на основе [прибора с зарядовой связью](https://ru.wikipedia.org/wiki/%D0%9F%D0%97%D0%A1) (детектор [Sony ICX424](https://s1-dl.theimagingsource.com/api/2.5/packages/publications/sensors-ccd/icx424al/e6f6a6dc-f966-5bf2-89ca-b1370715d416/icx424al_1.2.en_US.pdf)). В указанном файле расположен четырёхмерный массив данных, содержащий 100 пар изображений, каждое размером 659x493 пикселей. Каждая пара представляет из себя изображения, полученные с детектора, равномерно засвеченного с различными интенсивностями (всего 100 вариантов, внутри пары интенсивность одинаковая). Первая по порядку пара — результат мгновенного считывания, и представляет из себя сигнал детектора в отсутствии света.

Используя простые рассуждения, можно показать, что из таких данных возможно получить оценки _шума считывания_ (_read out noise_) и _коэффициента усиления_ (_gain_). Напомним, что каждый пиксель детектора накапливает заряд, производимый за счёт внутреннего фотоэлектрического эффекта. После оцифровки величина сигнала выражается в условных _цифровых единицах_ (ADU), которые и записаны в файле, и связана с изначальным числом электронов следующим соотношением:

$$x = \frac{N_e}{g}$$

где `g` — _коэффициент усиления_ измеряемый в $[e^- / ADU]$, а дисперсия величины

$$
\sigma_x^2 = \frac{\sigma_r^2}{g^2}+\frac{\sigma_{N_e}^2}{g^2}=\frac{\sigma_r^2}{g^2} + \frac xg
$$

где $\sigma_r$ — _шум считывания_, для удобства измеряется в количестве $e^-$, а последний переход — следствие того, что для количества фотоотсчетов действует закон распределения Пуассона, и $\sigma^2_{N_e}=N_e$.

На практике удобнее рассматривать поэлементную разность значений пикселей внутри пары кадров, тогда

$$\sigma^2_{\Delta x} = 2\frac{\sigma^2_r}{g^2}+\frac{2}{g}x$$

где оценку $\sigma^2_{\Delta x}$ можно получить с помощью выборочной дисперсии по всем пикселям поэлементной разности кадров пары, а оценку `x` можно получить с помощью выборочного среднего по всем пикселям двух кадров пары. При этом для первой пары (в отсутствии света) средний сигнал будет ненулевой, он соответствует _напряжению смещения_, и не обусловлен световым потоком; поэтому полученную для первой пары величину следует вычесть из всех полученных оценок. Подробности доступны в [1].

[1]: Steve B. Howell, "Handbook of CCD Astronomy", Cambridge University Press, 2006

**Дедлайн 13 февраля 2025 в 23:55**

Вы должны сделать следующее:

 - В модуле `lsp` реализуйте функцию `lstsq_ne(a, b)` реализующую подход линейных наименьших квадратов с использованием нормальных уравнений. `a` — матрица задачи размера `(N, M)`, `b` — вектор задачи размера `(N,)`.
Функция должна возвращать кортеж `(x, cost, var)`, где `x` — решение задачи наименьших квадратов размера `(M,)`, `cost` — квадрат нормы вектора невязок размера, `var` — матрица ошибок решения размера `(N,N)`.

 - В модуле `lsp` реализуйте функцию `lstsq_svd(a, b, rcond=None)` реализующую подход линейных наименьших квадратов с использованием вычислений на основ алгоритма сингулярного разложения. Результат работы функции и смысл параметров совпадают с определением функции `lsqsq_ne`. Параметр `rcond` принимает необязательный аргумент для регуляризации, если аргумент задан, то все сингулярные числа меньше чем `rcond * s_max` должны игнорироваться.

 - В модуле `lsp` реализуйте функцию `lstsq(a, b, method, **kwargs)`, которая принимает дополнительный параметр `method` с возможными значениями `ne` или `svd` и возвращает результат работы одной из двух соответствующих функций. С помощью аргумента `kwargs` должна быть реализована передача специфичных для каждого из алгоритмов параметров.

 - В файле `eval.py` проверьте работоспособность методов, используя известные свойства оценок, получаемых методом наименьших квадратов:
    * случайно сгенерируйте одну матрицу `A` размера 500x20, и один вектор параметров `x` размера 20;
    * сгенерируйте 10000 реализаций случайного вектора `b` (размера 500) с многомерным нормальным распределением, чьё среднее — `Ax`, а ковариационная матрица — диагональная с дисперсией 0.01;
    * используя метод наименьших квадратов (функцию `lstsq`, реализованную в предыдущем упражнении), получите 10000 оценок неизвестных параметров `x` задачи МНК с матрицей `A` и правой частью `b`;
    * убедитесь, что все компоненты полученных оценок вектора параметров распределены нормально, причём выборочное среднее соответствует истинному вектору `x`, а выборочные дисперсии соответствуют рассчитанным по формуле для ошибок;
    * изобразите на графике `norm.png` частотную гистограмму величины невязки и теоретическое распределение. Для этого удобно использовать функцию `norm.pdf` из пакета `scipy.stats`;
    * убедитесь, что величины невязки распределены в соответствии с распределением "Хи-квадрат" с соответствующим числом степеней свободы и масштабом;
    * изобразите на графике `chi2.png` частотную гистограмму величины невязки и теоретическое распределение. Для этого удобно использовать функцию `chi2.pdf` из пакета `scipy.stats`.

 - В файле `ccd.py` находится заготовка программы, принимающая путь к файлу с данными в качестве аргумента командной строки. Например, загрузите файл `ccd_v2.fits`:
  ```bash
  > python3 ccd.py ccd_v2.fits
  ccd_v2.fits
  ```
  > Для задания аргументов командной строки в Spyder используйте меню "Запуск" -> "Настройки для файла..." (`Ctrl+F6`) -> "Опции командной строки"

  Нужно модифицировать код `ccd.py` таким образом, чтобы реализовать следующий функционал обработки:
    * загрузите файл с данными и вычислите оценки $\sigma^2_{\Delta x}$ от `x`;
    * точками изобразите на графике `ccd.png` зависимость $\sigma^2_{\Delta x}$ от `x`;
    * используя метод наименьших квадратов (функцию `lstsq`, реализованную в предыдущем упражнении), получите оценку двух коэффициентов линейной зависимости;
    * изобразите прямой линией на том же графике `ccd.png` модельную зависимость $\sigma^2_{\Delta x}$ от `x`;
    * используя полученные оценки коэффициентов линейной зависимости, получите оценки значений величин _коэффициента усиления_ `g` и _шума считывания_ $\sigma_r$;
    * используя формулу оценки погрешности в косвенных измерениях, оцените ошибки определения _коэффициента усиления_ `g` и _шума считывания_ $\sigma_r$;
    * используя технику *bootstrap*, проверьте, насколько хорошо в данном случае работает формула оценки погрешности в косвенных измерениях — для этого сгенерируйте 10000 подвыборок с повторами такого же размера, как исходные данные, для каждой подвыборки оцените _коэффициент усиления_ `g` и _шум считывания_ $\sigma_r$, а затем рассчитайте их выборочные стандартные отклонения;
    * результат сохраните в файле `ccd.json` в следующем формате:
    ```
    {
        "ron": 12.34,
        "ron_err": 0.12,
        "ron_err_bootstrap": 0.11,
        "gain": 1.234,
        "gain_err": 0.0012,
        "gain_err_bootstrap": 0.0013
    }
    ```
