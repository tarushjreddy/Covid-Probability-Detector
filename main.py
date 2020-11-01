import pickle
from flask import Flask, render_template, request
app = Flask(__name__)

file = open('model.pkl', 'rb')
#
# print(file)

clf = pickle.load(file)

# file.close()


@app.route('/', methods=["GET", "POST"])
def hello_world():
    if request.method == "POST":
        mydic = request.form
        name = str(mydic['name'])
        age = int(mydic['age'])
        FEVER = int(mydic['fever'])
        diffBreath = int(mydic['nose'])
        bodyPain = int(mydic['bodypain'])
        Oxylow = int(mydic['air'])
        print(FEVER, Oxylow)
        print(request.form)
        inputfeat = [FEVER, bodyPain, diffBreath, diffBreath, Oxylow, age]
        infec = clf.predict([inputfeat])[0]
        print("this is the result", infec)
        return render_template('result.html', inff=infec*100, namee=name)

    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
