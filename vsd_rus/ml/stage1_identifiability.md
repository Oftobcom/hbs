# Тихий режим — только отчёты и графики
python -m ml.stage1_identifiability -v 0

# По умолчанию — базовые принты
python -m ml.stage1_identifiability

# Детальный лог
python -m ml.stage1_identifiability -v 2

# Отладка базовой точки (backward-compat)
python ml/stage1_identifiability.py debug
python ml/stage1_identifiability.py debug -v 2
