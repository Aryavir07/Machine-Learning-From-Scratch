# creating our own API
# create venv: python -m venv .venv
# flask-sqlalchemy for database
# export FLASK_APP=ownapi.py
# export FLASK_ENV=development
# flask run
# ctrl+c : to stop
from flask import Flask
app = Flask(__name__)


from flask_sqlalchemy import SQLAlchemy
db = SQLAlchemy(app)
app.config['SQLALCHEMY_DATABASE_URL'] = 'sqlite:///data.db'


#connecting to database
class Drinks(db.Model):
    id = db.Column(db.Integer, primary_key = True)
    name = db.Column(db.String(80), unique =True, nullable = False)
    description = db.Column(db.String(120))

    def __repr__(self): # for representation
        return f'{self.name}-{self.description}'


@app.route('/')
def index():
    return 'Hello!'

# making GET request

@app.route('/drinks')
def get_drinks():
    drinks =  Drinks.query.all()
    output = []
    for drink in drinks:
        drink_data = {'name': drink.name, 'description': drink.description}
        output.append(drink_data)
    return {'drinks': output}


@app.rout('/drinks/<id>')

def get_drinks(id):
    drink = Drink.query.get_or_404(id)
    return jsonify({'name':drink.name, 'descrip': drink.description})
    