from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from flask import Flask, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import text

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

def regression_training():
    totalRevenue = db.engine.connect().execute(text('CALL QuantitySaleByYear(:year)'), {'year': 2024}).all()
    revenue = [ float(item._asdict()['sub_total']) for item in totalRevenue]
    quantities = [ float(item._asdict()['total_quantity']) for item in totalRevenue]

    X = np.array(quantities).reshape((-1, 1))
    y = np.array(revenue)

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=123)

    model = LinearRegression()

    model.fit(X=X_train, y=y_train)

    r_sq = model.score(X=X_train, y=y_train)

    print(f"coefficient of determination: {r_sq}, intercept: {model.intercept_}, coeff: {model.coef_}")

    # y_pred_test = model.predict(X=X_test)

    # acc = r2_score(y_pred=y_pred_test, y_true=y_test)

    # print(acc)

    quantity_mean = np.array(quantities).mean()

    X_new = np.array([[quantity_mean]])

    y_pred = model.predict(X=X_new)

    revenue.append(int(y_pred[0]))
    
    return revenue



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