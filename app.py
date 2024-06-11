from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from flask import Flask, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import text
import random

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqldb://root@localhost/dw_obat'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

@dataclass
class DimDepo(db.Model):
    depo_org_id: Mapped[int] = db.Column(db.Integer, primary_key=True)
    depo_name: Mapped[str] = db.Column(db.String(10))
    sales: Mapped[list] = db.relationship('Sale', backref='dim_depo', lazy=True) 

@dataclass
class DimObat(db.Model):
    drug_id: Mapped[str] = mapped_column(primary_key=True)
    drug_name: Mapped[str]
      

@dataclass
class DimTime(db.Model):
    sk_time: Mapped[str] = mapped_column(primary_key=True)
    date: Mapped[str]
    day: Mapped[int]
    month: Mapped[int]
    month_name: Mapped[str]
    year: Mapped[int]
    kuartal: Mapped[int]

@dataclass
class Sale(db.Model):
    __tablename__ = 'fact_sale'
    id: Mapped[int] = mapped_column(primary_key=True)
    quantity: Mapped[int]
    tarif: Mapped[int]
    ppn: Mapped[int]
    sub_total: Mapped[int]
    depo_org_id: Mapped[int] = db.Column(db.Integer, db.ForeignKey('dim_depo.depo_org_id'))
    drug_id: Mapped[str]
    sk_time: Mapped[str]

def preprocessing(drug_df):
    drug_df['tanggal'] = pd.to_datetime(drug_df['tanggal'])
    drug_df = drug_df.groupby("tanggal")[['quantity','sub_total']].agg('sum')
    drug_df['tanggal'] = drug_df.index
    drug_df['tahun'] = drug_df['tanggal'].dt.year
    drug_df['bulan'] = drug_df['tanggal'].dt.month

    return drug_df


def regression_training():
    drug_df = pd.read_csv('datasets/tb_transaction_202406101206.csv')

    drug_df = preprocessing(drug_df)

    X = drug_df[['quantity']]
    y = drug_df['sub_total']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LinearRegression()

    model.fit(X_train, y_train)

    r_sq = model.score(X_train, y_train)

    print(f"coefficient of determination: {r_sq}, intercept: {model.intercept_}, coeff: {model.coef_}")

    y_pred_test = model.predict(X_test)

    acc_score = r2_score(y_true=y_test, y_pred=y_pred_test)
    mae = mean_absolute_error(y_true=y_test, y_pred=y_pred_test)

    print(f"R^2 Score: {acc_score}")
    print(f"Mean Absolute Error: {mae}")

    y_pred_total = model.predict(drug_df[['quantity']])
    y_pred_total = np.array([ round(y_pred,0) for y_pred in y_pred_total])

    drug_df['sub_total_predict'] = y_pred_total

    previous_sub_total_df = drug_df[drug_df['bulan'] == 5]
    total_data = previous_sub_total_df.count()[0]
    min_quantity = previous_sub_total_df['quantity'].min()
    max_quantity = previous_sub_total_df['quantity'].max()

    new_data = []
    for _ in range(total_data):
        new_data.append(random.randint(min_quantity, max_quantity))

    new_data = np.array(new_data).reshape(-1,1)

    y_pred = model.predict(new_data)
    y_pred = np.array([ round(y,0) for y in y_pred])
    
    result_df = pd.DataFrame({
        'bulan': drug_df['bulan'].unique(),
        'quantity': drug_df.groupby('bulan')[['quantity']].agg('sum')['quantity'],
        'sub_total_true': drug_df.groupby('bulan')[['sub_total']].agg('sum')['sub_total'],
        'sub_total_predict': drug_df.groupby('bulan')[['sub_total_predict']].agg('sum')['sub_total_predict']
    })

    revenue = result_df[['sub_total_true']].values.reshape(-1,).tolist()

    revenue.append(y_pred.sum())

    return revenue
    # totalRevenue = db.engine.connect().execute(text('CALL QuantitySaleByYear(:year)'), {'year': 2024}).all()
    # revenue = [ float(item._asdict()['sub_total']) for item in totalRevenue]
    # quantities = [ float(item._asdict()['total_quantity']) for item in totalRevenue]

    # X = np.array(quantities).reshape((-1, 1))
    # y = np.array(revenue)

    # X_train, X_test, y_train, y_test = train_test_split(
    # X, y, test_size=0.4, random_state=123)

    # model = LinearRegression()

    # model.fit(X=X_train, y=y_train)

    # r_sq = model.score(X=X_train, y=y_train)

    # print(f"coefficient of determination: {r_sq}, intercept: {model.intercept_}, coeff: {model.coef_}")

    # # y_pred_test = model.predict(X=X_test)

    # # acc = r2_score(y_pred=y_pred_test, y_true=y_test)

    # # print(acc)

    # quantity_mean = np.array(quantities).mean()

    # X_new = np.array([[quantity_mean]])

    # y_pred = model.predict(X=X_new)

    # revenue.append(int(y_pred[0]))
    
    # return revenue



@app.get("/")
def dashboard():
    # drugs = db.engine.connect().execute(text('select * from dim_obat')).all()
    # depos = db.engine.connect().execute(text('select * from dim_depo')).all()
    # allDepo = [ item._asdict()  for item in depos]
    # data = [ item._asdict()  for item in drugs]

    
    return render_template('dashboard.html')

@app.get('/api/depos')
def allDepo():
    depos = db.engine.connect().execute(text('select * from dim_depo')).all()
    data = [ item._asdict()  for item in depos]
    return jsonify({'data': data})

@app.get('/api/drugs')
def allDrug():
    drugs = db.engine.connect().execute(text('select * from dim_obat')).all()
    data = [ item._asdict()  for item in drugs]
    return jsonify({'data': data})

@app.get('/api/dashboard/drugs')
def getSalesAndQuantitiesOfDrugs():
    salesAndQuantitiesOfDrugs = db.engine.connect().execute(text('CALL QuantitySaleDrugByYear(:year)'), {'year': 2024}).all()
    drugOne = []
    drugTwo = []
    quantityDrugOne = []
    quantityDrugTwo = []
    for item in salesAndQuantitiesOfDrugs:
        drug = item._asdict()
        if drug['drug_id'] == 'DRx0014697':
            drugOne.append(drug['sub_total'])
            quantityDrugOne.append(drug['total_quantity'])
        if drug['drug_id'] == 'DRx0014992':
            drugTwo.append(drug['sub_total'])
            quantityDrugTwo.append(drug['total_quantity'])
    revenueDrugs = [
        {
            'label': 'Obat A',
            'data': drugOne
        },
        {
            'label': 'Obat B',
            'data': drugTwo
        }
    ]

    quantityDrugs = [
        {
            'label': 'Obat A',
            'data': quantityDrugOne
        },
        {
            'label': 'Obat B',
            'data': quantityDrugTwo
        }
    ]
    return jsonify({'data': {'revenue': revenueDrugs, 'quantities': quantityDrugs}})

@app.get('/api/dashboard/depos')
def getSalesAndQuantitiesOfDepos():
    # depoIds = [1, 45, 70, 109, 119, 170, 218, 401, 404, 415]
    # revenueDepos = [
    #     {
    #         'label': 'Depo A',
    #         'data': []
    #     },
    #     {
    #         'label': 'Depo I',
    #         'data': []
    #     },
    #     {
    #         'label': 'Depo J',
    #         'data': []
    #     },
    #     {
    #         'label': 'Depo B',
    #         'data': []
    #     },
    #     {
    #         'label': 'Depo C',
    #         'data': []
    #     },
    #     {
    #         'label': 'Depo D',
    #         'data': []
    #     },
    #     {
    #         'label': 'Depo E',
    #         'data': []
    #     },
    #     {
    #         'label': 'Depo F',
    #         'data': []
    #     },
    #     {
    #         'label': 'Depo G',
    #         'data': []
    #     },
    #     {
    #         'label': 'Depo H',
    #         'data': []
    #     },
    # ]
    salesAndQuantitiesOfDepos = db.engine.connect().execute(text('CALL QuantitySaleDepoByYear(:year)'), {'year': 2024}).all()
    # for item in salesAndQuantitiesOfDepos:
    #     depo = item._asdict()
    #     if depo['depo_org_id'] == 1:
    #         revenueDepos[0]['data'].append(depo['sub_total'])
    #     elif depo['depo_org_id'] == 45:
    #         revenueDepos[1]['data'].append(depo['sub_total'])
    #     elif depo['depo_org_id'] == 70:
    #         revenueDepos[2]['data'].append(depo['sub_total'])
    #     elif depo['depo_org_id'] == 109:
    #         revenueDepos[3]['data'].append(depo['sub_total'])
    #     elif depo['depo_org_id'] == 119:
    #         revenueDepos[4]['data'].append(depo['sub_total'])
    #     elif depo['depo_org_id'] == 170:
    #         revenueDepos[5]['data'].append(depo['sub_total'])
    #     elif depo['depo_org_id'] == 218:
    #         revenueDepos[6]['data'].append(depo['sub_total'])
    #     elif depo['depo_org_id'] == 401:
    #         revenueDepos[7]['data'].append(depo['sub_total'])
    #     elif depo['depo_org_id'] == 404:
    #         revenueDepos[8]['data'].append(depo['sub_total'])
    #     elif depo['depo_org_id'] == 415:
    #         revenueDepos[9]['data'].append(depo['sub_total'])

    revenueDepos = [ item._asdict()['sub_total']  for item in salesAndQuantitiesOfDepos]
    labels = [ item._asdict()['depo_name']  for item in salesAndQuantitiesOfDepos]


    return jsonify({'data': {'revenue': {'label': labels, 'data': revenueDepos}}})

@app.get('/api/dashboard/total-revenue')
def getTotalRevenue():
    totalRevenue = db.engine.connect().execute(text('CALL QuantitySaleByYear(:year)'), {'year': 2024}).all()
    revenue = [ item._asdict()['sub_total'] for item in totalRevenue]
    return jsonify({'data': {'revenue': revenue}})

@app.get('/api/dashboard/revenue-predict')
def test():
    revenue_predict = regression_training()
    return jsonify({'data': {'revenue': revenue_predict}})