# cognigraph
Inverse-modelling-related capabilities of cognigraph

## Инструкции по установке
1. **Питон и пакеты.** 
Самый простой вариант - через среду conda. 

```bash
conda create -n cognigraph python=3 pyqt=5 pyqtgraph ipython scipy numba sympy sklearn pandas matplotlib numba
activate cognigraph
pip install pylsl expyriment mne
```

**Осторожно!**
Возможны проблемы с версией python 3.7 в связи с багами в пакете cython.
Рекомендуется использовать версию питона 3.6 или более раннюю.

2. **Репозиторий.** Часть зависимостей организована через подмодули git. Для
того, чтобы они загрузились вместе с текущим репозиторием при клонировании 
необходимо добавить флаг `--recursive`.
Далее необходимо перейти в папку репозитория и установить пакет:

```bash
git clone --recursive git@github.com:dmalt/cognigraph.git
cd cognigraph
pip install --editable .
```


3. **Необходимые файлы.** Программа использует файлы из датасета _sample_, 
распространяемого с пакетом _mne-python_. Чтобы не качать все файлы (датасет
лежит на osf.io, загрузка с которого  происходит крайне медленно), можно скачать
урезанную версию 
[отсюда](https://drive.google.com/open?id=1D0jI_Z5EycI8JwJbYOAYdSycNGoarmP-). 
Папку _MNE-sample-data_ из архива надо скопировать в то же место, куда бы ее 
загрузил _mne-python_. Чтобы узнать, что это за место, не скачивая датасет, 
нужно сделать следующее: 

    ```
    from mne.datasets import sample
    print(sample.data_path(download=False, verbose=False))
    ```
    Папку _MNE-sample-data_ из архива копируем в выведенный путь.



