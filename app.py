from flask import Flask, render_template, redirect, request
from src.pipelines.Models import model_dict
from src.helper import datasets_path, json_load

app = Flask(__name__)



@app.route('/')
def home():
    return render_template('home.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')




@app.route('/explore_datasets')
def explore_datasets():
    return render_template('explore_datasets.html', dataset_dict = datasets_path(), count = enumerate)


@app.route('/explore_models')
def explore_models():
    return render_template('explore_models.html', model_dict=model_dict, count = enumerate)


@app.route('/explore_categories')
def explore_categories():
    cat = json_load('artifacts/def_types.json')
    return render_template('explore_categories.html', categories = datasets_path().keys(), count = enumerate, cat = cat)





@app.route('/model/<model>')
def model(model):
    types = request.args.get('types')
    return render_template('model.html', name=model, types=types)


@app.route('/dataset/<dataset>')
def dataset(dataset):
    types = request.args.get('types')
    return render_template('dataset.html', name=dataset, types=types)


@app.route('/category/<category>')
def category(category):
    types = request.args.get('types')
    return render_template('category.html', name=category, types=types)


@app.route('/explore')
def explore():
    option = request.args.get('option')      ## eg: 'explore_datasets' 
    name = request.args.get('name')          ## eg: 'iris' or 'classification'
    which = request.args.get('which')        ## eg: is it dataset or model
    types = request.args.get('types')        ## eg: 'classification' or 'regression'
    return redirect(f"http://3.92.136.49:8501/?page={which}&select={option}&name={name}&types={types}")

    return redirect(f"http://localhost:8501/?page={which}&select={option}&name={name}&types={types}")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)