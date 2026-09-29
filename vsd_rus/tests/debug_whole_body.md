# По умолчанию — scenario vsd_r5 (тот же, что в Stage 1)
python debug_whole_body.py

# Всё сразу — увидеть, как меняется картина от здорового к большому ДМЖП
python debug_whole_body.py --variant all

# Только здоровый — понять базовую линию
python debug_whole_body.py --variant healthy

# Только большой ДМЖП
python debug_whole_body.py --variant vsd_r1

# Синдром Эйзенменгера
python debug_whole_body.py --variant eisenmenger
