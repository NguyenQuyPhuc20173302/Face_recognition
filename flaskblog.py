import shutil
import take_data
from flask import Flask, render_template, url_for, flash, redirect, request
from forms import RegistrationForm, LoginForm, Data, user
import os
from time import time
import face_train
import face_detection_video

app = Flask(__name__)
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'
UN = ''
us = user()
posts = [
    {
        'author': 'Xin chúc mừng',
        'title': 'Dữ liệu của bạn đang được xử lý',
        'content': 'Vui lòng chờ trong giây lát...',
        'date_posted': 'bạn đã thêm dữ liệu thành công'
    }

]
posts1 = [
    {
        'author': 'Hoàn tất',
        'title': '',
        'content': '',
        'date_posted': ''
    }

]


@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html')


@app.route("/home1")
def home1():
    face_train.traning()
    return render_template('home1.html', posts=posts)


@app.route("/training")
def training():
    face_train.traning()
    return render_template('home.html')


@app.route("/faceRecognition")
def faceRecognition():
    face_detection_video.face_R()
    return render_template('home.html')


@app.route("/register", methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        flash(f'Account created for {form.username.data}!', 'success')
        return redirect(url_for('home'))
    return render_template('register.html', title='Register', form=form)


@app.route("/login", methods=['GET', 'POST'])
def login():
    form = LoginForm()

    if form.validate_on_submit():
        if form.email.data == 'nguyenquyphuc1591999@gmail.com' and form.password.data == '1':
            flash('You have been logged in!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Login Unsuccessful. Please check username and password', 'danger')
    return render_template('login.html', title='Login', form=form)


@app.route("/addData", methods=['GET', 'POST'])
def addData():
    form = Data()
    if form.validate_on_submit():
        UN = form.username.data
        us.username.data = UN
        if len(UN) > 0:
            f = 'image/' + UN
            try:
                os.mkdir(f)
            except:
                return render_template('addData.html', title='ADD', form=form)
            return render_template('take_image1.html', title='ADD', name=UN)
    return render_template('addData.html', title='ADD', form=form)


@app.route("/deleteData", methods=['GET', 'POST'])
def deleteData():
    form = Data()
    if form.validate_on_submit():
        name = form.username.data
        if len(name) > 0:
            f = 'image/' + name
            try:
                shutil.rmtree(f)
            except:
                return render_template('deleteData.html', title='DELETE', form=form)
            return render_template('home.html', title='ADD')
    return render_template('deleteData.html', title='ADD', form=form)


@app.route("/take_image1")
def take_image1():
    us1 = user()
    us1.username.data = us.username.data
    name = us1.username.data
    take_data.take_image(name)
    return redirect(url_for('home1'))


if __name__ == '__main__':
    app.run(debug=True)
