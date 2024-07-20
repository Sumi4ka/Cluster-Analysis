#K-means Clustering on Iris Dataset

Этот проект реализует алгоритм кластеризации k-means (k-средних, алгоритм Ллойда) для набора данных Iris. K-means — это метод кластеризации, который группирует данные в 
K кластеров, минимизируя сумму квадратов расстояний между точками и центрами их кластеров. Программа выполняет многократную инициализацию центров кластеров, оценивает качество кластеризации и визуализирует результаты.
Выбран датасет iris.


##Требования:

Необходимо установить для запуска  python с официального сайта и VScode с плагином Python.

Для запуска программы необходимо установить следующие библиотеки:

numpy
scikit-learn
matplotlib
Вы можете установить их с помощью pip:

        pip install numpy scikit-learn matplotlib

###Для запуска пропишите в терминале данную строку:        

        python clustering.py

##Описание работы

###Инициализация центров кластеров:

    Случайным образом выбираются 𝐾 к начальных центров кластеров.
###Цикл до сходимости:

    Обновление меток кластеров на основе текущих центров (Expectation).
    Пересчет центров кластеров как средних значений точек, принадлежащих каждому кластеру (Maximization).
    Проверка сходимости по изменению центров кластеров.

###Оценка и сохранение результата:

    Оценка качества кластеризации по сумме расстояний от точек до центров их кластеров.
    Повторение процесса M раз для улучшения результатов за счет разной инициализации центров кластеров.
###Визуализация:

    Визуализация кластеров с использованием первых двух признаков данных Iris.        
