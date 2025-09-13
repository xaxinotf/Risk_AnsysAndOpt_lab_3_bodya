# Критерії прийняття рішень у ризику — Dash демо

- Редагована матриця виграшів і ймовірності станів.
- Критерії: Бернуллі–Лаплас, Гурвіц, Ходжес–Леман, Мінімум дисперсії.
- Візуалізації: теплокарта, стовпчики, чутливість до α.
- Рекомендація стратегії за обраним критерієм.

## Запуск
```bash
cd /mnt/data/decision_criteria_dash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```
Відкрийте http://127.0.0.1:8050
