from datetime import datetime
import os
from werkzeug.utils import secure_filename, send_file

from utils.image_predictor import predict_image
from flask import Flask,render_template,request,redirect,url_for,flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager,UserMixin,login_user,logout_user,login_required,current_user
from werkzeug.security import generate_password_hash,check_password_hash
from utils.predictor import predict_news
from reportlab.pdfgen import canvas

app=Flask(__name__)
app.config["SECRET_KEY"] = "deepfake123"

app.config["SQLALCHEMY_DATABASE_URI"]="sqlite:///database.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"]=False

UPLOAD_FOLDER="uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"]=UPLOAD_FOLDER

db=SQLAlchemy(app)

login_manager=LoginManager()
login_manager.login_view="login"
login_manager.init_app(app)

class User(UserMixin,db.Model):
    id=db.Column(db.Integer,primary_key=True)
    username=db.Column(db.String(100),unique=True)
    email=db.Column(db.String(120),unique=True)
    password=db.Column(db.String(200))

class History(db.Model):

    id=db.Column(db.Integer,primary_key=True)

    username=db.Column(db.String(100))

    prediction=db.Column(db.String(50))

    confidence=db.Column(db.Float)

    filename=db.Column(db.String(200))

    created=db.Column(db.DateTime,default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

@app.route("/")
def home():
    return render_template("index.html")
@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        username = request.form["username"]
        email = request.form["email"]
        password = generate_password_hash(request.form["password"])

        # Check existing username
        if User.query.filter_by(username=username).first():
            flash("Username already exists!")
            return redirect(url_for("register"))

        # Check existing email
        if User.query.filter_by(email=email).first():
            flash("Email already registered!")
            return redirect(url_for("register"))

        user = User(
            username=username,
            email=email,
            password=password
        )

        db.session.add(user)
        db.session.commit()

        flash("Registration Successful")
        return redirect(url_for("login"))

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form["email"]
        password = request.form["password"]

        print("Email:", email)

        user = User.query.filter_by(email=email).first()

        print("User:", user)

        if user:
            print("Password Match:", check_password_hash(user.password, password))

        if user and check_password_hash(user.password, password):

            login_user(user)

            print("Logged In:", current_user.is_authenticated)

            return redirect(url_for("dashboard"))

        flash("Invalid Credentials")

    return render_template("login.html")

@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html",name=current_user.username)

@app.route("/profile")
@login_required
def profile():

    return render_template(

        "profile.html",

        user=current_user

    )

@app.route("/admin")
@login_required
def admin():

    total=History.query.count()

    fake=History.query.filter(

        History.prediction.like("%FAKE%")

    ).count()

    real=History.query.filter(

        History.prediction.like("%REAL%")

    ).count()

    return render_template(

        "admin.html",

        total=total,

        fake=fake,

        real=real

    )

@app.route("/logout")
@login_required
def logout():

    logout_user()

    return redirect(url_for("home"))

@app.route("/history")
@login_required
def history():

    data = History.query.filter_by(
        username=current_user.username
    ).order_by(
        History.created.desc()
    ).all()

    return render_template("history.html", history=data)

from reportlab.pdfgen import canvas

@app.route("/report")
@login_required

def report():

    pdf=canvas.Canvas("report.pdf")

    pdf.drawString(

        100,

        750,

        "Deepfake Detection Report"

    )

    pdf.save()

    return send_file(

        "report.pdf",

        as_attachment=True

    )

@app.route("/detect-text")
@login_required
def detect_text():

    return render_template("detect_text.html")

@app.route("/predict-text", methods=["POST"])
@login_required
def predict_text():

    news = request.form["news"]

    result, confidence = predict_news(news)

    # Save prediction history
    history = History(
        username=current_user.username,
        prediction=result,
        confidence=confidence,
        filename="Text News"
    )

    db.session.add(history)
    db.session.commit()

    return render_template(
        "result.html",
        result=result,
        confidence=confidence,
        news=news
    )
@app.route("/detect-image")
@login_required
def detect_image():

    return render_template("detect_image.html")

@app.route("/predict-image", methods=["POST"])
@login_required
def predict_image_route():

    file = request.files["image"]

    filename = secure_filename(file.filename)

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    file.save(filepath)

    result, confidence = predict_image(filepath)

    # Save prediction history
    history = History(
        username=current_user.username,
        prediction=result,
        confidence=confidence,
        filename=filename
    )

    db.session.add(history)
    db.session.commit()

    return render_template(
        "image_result.html",
        result=result,
        confidence=confidence,
        image=filename
    )
if __name__=="__main__":

    with app.app_context():
        db.create_all()

    app.run(debug=True)