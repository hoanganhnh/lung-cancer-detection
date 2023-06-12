import os
from cs50 import SQL
from flask import Flask, flash, redirect, render_template, request, session, url_for
from flask_session import Session
from tempfile import mkdtemp
from werkzeug.exceptions import default_exceptions, HTTPException, InternalServerError
from werkzeug.security import check_password_hash, generate_password_hash
from datetime import datetime
import cv2
import numpy as np

from helpers import apology, login_required
import segmentation
from model.model import predictImg, readImage
from text_process import text_predict_cancer

# Configure application
app = Flask(__name__, static_url_path='/static')
app.config['SESSION_TYPE'] = 'memcached'
app.config['SECRET_KEY'] = 'super secret key'
sess = Session()

# Ensure templates are auto-reloaded
app.config["TEMPLATES_AUTO_RELOAD"] = True

# Configure CS50 Library to use SQLite database
db = SQL("sqlite:///data.db")

# db.execute("ALTER TABLE scans DROP COLUMN annotated_scan")
# db.execute("ALTER TABLE scans DROP COLUMN area")

# db.execute(
#     "ALTER TABLE scans ADD predict varchar(255)")
# print(db.execute("DELETE FROM scans WHERE date < '2023-01-01 00:00:00'"))


@app.after_request
def after_request(response):
    """Ensure responses aren't cached"""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response


@app.route("/")
@login_required
def index():
    """Show list of patients"""
    # Retrieve all patients assigned to doctor
    patients = db.execute(
        "SELECT name, dob FROM patients WHERE doctor_id = ?", session["user_id"])

    return render_template("index.html", patients=patients)


@app.route("/add", methods=["GET", "POST"])
@login_required
def add():
    """Add new patient"""

    # User reached route via POST (as by submitting a form via POST)
    if request.method == "POST":

        # Ensure name was submitted
        if not request.form.get("name"):
            return apology("must provide patient name", 400)

        # Ensure DOB was submitted
        if not request.form.get("dob"):
            return apology("must provide patient DOB", 400)

        # Query database for patient name
        rows = db.execute(
            "SELECT name FROM patients WHERE name = ?", request.form.get("name"))

        # Ensure patient does not already exist
        if len(rows) == 1:
            return apology("username already exists", 400)

        # Update patient list
        db.execute("INSERT INTO patients (name, dob, doctor_id) VALUES (?, ?, ?)",
                   request.form.get("name"), request.form.get("dob"), session["user_id"])

        # Redirect user to home page
        return redirect("/")

    # User reached route via GET (as by clicking a link or via redirect)
    else:
        return render_template("add.html")


@app.route("/delete", methods=["GET", "POST"])
@login_required
def delete():
    """Delete patient from records"""

    # User reached route via POST (as by submitting a form via POST)
    if request.method == "POST":

        # Ensure symbol was submitted
        if not request.form.get("name"):
            return apology("must provide patient name", 400)

        # Update patient list
        db.execute("DELETE FROM patients WHERE name = ?",
                   request.form.get("name"))

        # Redirect user to home page
        return redirect("/")

    # User reached route via GET (as by clicking a link or via redirect)
    else:
        patients = db.execute(
            "SELECT name FROM patients WHERE doctor_id = ?", session['user_id'])
        return render_template("delete.html", patients=patients)


@app.route("/login", methods=["GET", "POST"])
def login():
    """Log user in"""

    # Forget any user_id
    session.clear()

    # User reached route via POST (as by submitting a form via POST)
    if request.method == "POST":

        # Ensure username was submitted
        if not request.form.get("username"):
            return apology("must provide username", 400)

        # Ensure password was submitted
        elif not request.form.get("password"):
            return apology("must provide password", 400)

        # Query database for username
        rows = db.execute("SELECT * FROM doctors WHERE user = ?",
                          request.form.get("username"))

        # Ensure username exists and password is correct
        if len(rows) != 1 or not check_password_hash(rows[0]["pass"], request.form.get("password")):
            return apology("invalid username and/or password", 400)

        # Remember which user has logged in
        session["user_id"] = rows[0]["id"]

        # Redirect user to home page
        return redirect("/")

    # User reached route via GET (as by clicking a link or via redirect)
    else:
        return render_template("login.html")


@app.route("/logout")
def logout():
    """Log user out"""

    # Forget any user_id
    session.clear()

    # Redirect user to login form
    return redirect("/")


@app.route("/register", methods=["GET", "POST"])
def register():
    """Register user"""

    # Forget any user_id
    session.clear()

    # User reached route via POST (as by submitting a form via POST)
    if request.method == "POST":

        # Ensure username was submitted
        if not request.form.get("username"):
            return apology("must provide username", 400)

        # Ensure password was submitted
        elif not request.form.get("password"):
            return apology("must provide password", 400)

        # Ensure confirmation was submitted
        elif not request.form.get("confirmation"):
            return apology("must confirm password", 400)

        # Ensure password matches confirmation
        elif not (request.form.get("password") == request.form.get("confirmation")):
            return apology("password and confirmation must match", 400)

        # Query database for username
        rows = db.execute("SELECT * FROM doctors WHERE user = ?",
                          request.form.get("username"))

        # Ensure username does not already exist
        if len(rows) == 1:
            return apology("username already exists", 400)

        # Generate password hash
        hash = generate_password_hash(request.form.get(
            "password"), method='pbkdf2:sha256', salt_length=8)

        db.execute("INSERT INTO doctors (user, pass) VALUES (?, ?)",
                   request.form.get("username"), hash)

        # Take user to login page
        return render_template("login.html")

    # User reached route via GET (as by clicking a link or via redirect)
    else:
        return render_template("register.html")


@app.route("/select-patient", methods=["GET", "POST"])
@login_required
def select_patient():
    """Select patient for which to display scan"""

    # User reached route via POST (as by submitting a form via POST)
    if request.method == "POST":

        # Ensure symbol was submitted
        if not request.form.get("name"):
            return apology("must provide patient name", 400)

        # pull list of PET scan dates belonging to patient of interest
        dates = db.execute(
            "SELECT date FROM scans WHERE patient_id = (SELECT id FROM patients WHERE name = ?) ORDER BY date DESC", request.form.get("name"))

        # store patient name so it is accessible by select_scan method
        session["patient_name"] = request.form.get("name")

        # Redirect user to 'select-scan' page to select date of specific scan of interest
        return render_template("select-scan.html", patient=request.form.get("name"), dates=dates)

    # User reached route via GET (as by clicking a link or via redirect)
    else:
        patients = db.execute(
            "SELECT name FROM patients WHERE doctor_id = ?", session['user_id'])
        return render_template("select-patient.html", patients=patients)


@app.route("/select-scan", methods=["GET", "POST"])
@login_required
def select_scan():
    """Select date for which to display scan"""

    # User reached route via POST (as by submitting a form via POST)
    if request.method == "POST":

        # Ensure symbol was submitted
        if not request.form.get("date"):
            return apology("must provide date of scan", 400)

        # pull relevant PET scan information
        scan = db.execute("SELECT date, raw_scan, predict FROM scans WHERE date = ? AND patient_id = (SELECT id FROM patients WHERE name = ?)  ORDER BY date DESC",
                          request.form.get("date"), session["patient_name"])[0]

        # Write raw scan image to project directory
        nparr = np.frombuffer(scan['raw_scan'], np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        cv2.imwrite('./static/images/raw_scan.jpg', img)

        # Redirect user to 'display-scan' page to show PET scan data
        return render_template("display-scan.html", patient=session["patient_name"], date=scan['date'], predict=scan['predict'])

    # User reached route via GET (as by clicking a link or via redirect)
    else:
        return render_template("select-scan.html")


@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    """Upload new scan for patient"""

    # User reached route via POST (as by submitting a form via POST)
    if request.method == "POST":

        # Ensure name was submitted
        if not request.form.get("name"):
            return apology("must provide patient name", 400)

        # Ensure scan was submitted
        if request.files["file"].filename == '':
            return apology("must provide patient CT scan", 400)

        # Find patient ID corresponding to provided name
        patient_id = db.execute(
            "SELECT id FROM patients WHERE name = ?", request.form.get("name"))[0]['id']

        # Find current date and time
        date = str(datetime.now())

        PATH_IMG = './static/images/raw_scan.jpg'

        # Save scan as .jpg image to project directory
        request.files['file'].save(PATH_IMG)

        # Convert raw scan to compatible format for database
        raw_scan = cv2.imencode('.jpg', cv2.imread(PATH_IMG))[1].tobytes()

        raw_img = readImage(PATH_IMG)
        predict = predictImg(raw_img)

        # Insert all PET scan data into scans table
        db.execute("INSERT INTO scans (patient_id, raw_scan, predict, date) VALUES (?, ?, ?, ?)",
                   patient_id, raw_scan, predict, date)

        # Redirect user to 'display-scan' page to show PET scan data
        return render_template("display-scan.html", patient=request.form.get("name"), date=date, predict=predict)

    # User reached route via GET (as by clicking a link or via redirect)
    else:
        patients = db.execute(
            "SELECT name FROM patients WHERE doctor_id = ?", session["user_id"])
        return render_template("upload.html", patients=patients)


@app.route("/input-rqm", methods=["GET", "POST"])
@login_required
def input_rqm():
    if request.method == "POST":
        if not request.form.get("gender"):
            return apology("must provide gender", 400)

        gender = int(request.form.get("gender"))
        age_group = int(request.form.get("age_group"))
        farmer = int(request.form.get("farmer"))
        distance_from_crop = int(request.form.get("distance_from_crop"))
        air_pollution = int(request.form.get("air_pollution"))
        cooking = int(request.form.get("cooking"))
        polluted_environment = int(request.form.get("polluted_environment"))
        smoking = int(request.form.get("smoking"))
        number_smoked = int(request.form.get("number_smoked"))
        pesticides = int(request.form.get("pesticides"))
        herbicides = int(request.form.get("herbicides"))

        result = text_predict_cancer.predict(
            gender, age_group, farmer, distance_from_crop, cooking, air_pollution, cig_smoke=number_smoked, herbicide=herbicides, insecticides=pesticides)

        return render_template("display-search.html", predict=result)
    else:
        return render_template("input-rqm.html")


@app.route("/height-prediction", methods=["GET", "POST"])
@login_required
def height_prediction():
    if request.method == "POST":
        return render_template("display-height-prediction.html", height=165)
    else:
        return render_template("height-prediction.html")


def errorhandler(e):
    """Handle error"""
    if not isinstance(e, HTTPException):
        e = InternalServerError()
    return apology(e.name, e.code)


# Listen for errors
for code in default_exceptions:
    app.errorhandler(code)(errorhandler)
