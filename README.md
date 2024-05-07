# Accurate Captcha Solver

Solving captchas with highest accuracy approx 99% using HOG

## Get Started

1. Create virtual environment and activate it

```bash
python3 -m venv .venv && source .venv/bin/activate
```

2. Install Dependencies

```bash
pip install -r requirements.txt
```

3. Train your model (Run train script from inside src directory or change data directory from "../data" to just "data")

```bash
cd src && python3 Training.py
```

4. Predict results on your custom dataset

```bash
python3 Prediction.py
```

## Thank You (You made it possible)

A custom OCR built with HOG and logistic regression
![alt text](result.png)

Demo video ![ALPR](https://youtu.be/a_9utMqv2u4)
