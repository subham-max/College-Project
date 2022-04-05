import webbrowser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model


####LOAD DATA####
data=pd.read_csv('World_Data_Set.csv',sep=',')
data=data[['Days','cases']]
print('-'*30);print('HEAD');print('-'*30)
print(data.head())

####PREPARE DATA####
print('-'*30);print('PREPARE DATA');print('-'*30)
x=np.array(data['Days']).reshape(-1,1)
y=np.array(data['cases']).reshape(-1,1)
plt.plot(y,'-m')
#plt.show()
polyFeat=PolynomialFeatures(degree=2)
x=polyFeat.fit_transform(x)
#print(x)



####TRAINING DATA####
print('-'*30);print('TRAINING DATA');print('-'*30)
model=linear_model.LinearRegression()
model.fit(x,y)
accuracy=model.score(x,y)
printaccuracy=round(accuracy*100,3)
y0=model.predict(x)



####PREDICTION####
days=2
print('-'*30);print('PREDICTION');print('-'*30)
print(f'Prediction - Cases after {days} days:',end='')
printprediction = round(int(model.predict(polyFeat.fit_transform([[234+days]])))/1000000,2)

x1=np.array(list(range(1,234+days))).reshape(-1,1)
y1=model.predict(polyFeat.fit_transform(x1))
plt.plot(y1,'--r')
plt.plot(y0,'--b')
# plt.show()

html_content = f"<html><head>" \
               f"<title>" \
               f"Covid19 | Coronavirus (COVID-19) Prevention & Informatics</title>" \
               f"<script src='https://cdnjs.cloudflare.com/ajax/libs/Chart.js/2.5.0/Chart.min.js'></script>" \
               f"<link href='https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css' rel='stylesheet' integrity='sha384-1BmE4kWBq78iYhFldvKuhfTAU6auU8tT94WrHftjDbrCEXSU1oBoqyl2QvZ6jIW3' crossorigin='anonymous'>" \
               f"<script src='https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js' integrity='sha384-ka7Sk0Gln4gmtz2MlQnikT1wXgYsOg+OMhuP+IlRH9sENBO0LRn5q+8nbTov4+1p' crossorigin='anonymous'></script>" \
               f"<link rel='shortcut icon' href='img/coronavirus-5107715_1280.webp'></head>" \
               f"<body>" \
               f"<div class='container'> <header class='d-flex flex-wrap align-items-center justify-content-center justify-content-md-between py-3 mb-4 border-bottom'> <a href='#' class='d-flex align-items-center col-md-3 mb-2 mb-md-0 text-dark text-decoration-none'> <img src='img/coronavirus_logo-2.jpg' width='180' height='60'> </a>  <ul class='nav col-12 col-md-auto mb-2 justify-content-center mb-md-0'><h5><li><a href='#' class='nav-link px-2 link-secondary'>Home</a></li></h5><h5><a href='#about' class='nav-link px-2 link-dark'>About Corona</a></li></h5><h5><a href='#symptoms' class='nav-link px-2 link-dark'>Symtoms</a></li></h5><h5><a href='#Prevention' class='nav-link px-2 link-dark'>Prevention</a></li></h5><h5><a href='#graph' class='nav-link px-2 link-dark'>Graph</a></li></h5></ul> <div class='col-md-3 text-end'><button type='button' class='btn btn-outline-primary me-2'>Do & Don't</button></div></header></div>"\
               f"<div class='nk-banner'><div class='container'> <div class='row g-gs align-items-center justify-content-between'> <div class='col-lg-5 order-lg-last'> <div> <img src='img/header-a.png' width='500' height='450' alt=''></div></div><div class='col-lg-6'> <div class='nk-banner-block'> <div class='content'> <h1 class='heading'>CORONA <font style='color : Red;'><b>COVID-19</b></font> VIRUS</h1> <p>The Coronavirus (<font style='color : Red;'><b>COVID-19</b></font>) was first reported in Wuhan, Hubei, China in December 2019, the outbreak was later recognized as a pandemic by the World Health Organization (WHO) on 11 March 2020.</p><button type='button' class='btn btn-outline-primary me-2'>How to Protect</button> <button type='button' class='btn btn-outline-primary me-2'>About <font style='color : Red;'><b>COVID-19</b></font></button> <br><br><br><div class='status' data-covid='world'> <div class='row g-gs'> <div class='col-sm-4 col-6'> <div class='status-item'> <h6 class='title'>Worldwide Cases</h6> <div class='h3 count covid-stats-recovered'> <font style='color : blue;'>445,476,437</font></div></div></div><div class='col-sm-4 col-6'> <div class='status-item'> <h6 class='title'>Deaths</h6> <div class='h3 count covid-stats-recovered'><font style='color : red;'>116,654,456</font></div></div></div><div class='col-sm-4 col-6'> <div class='status-item'> <h6 class='title'>Recovered</h6> <div class='h3 count covid-stats-recovered'><font style='color : green;'>325,564,879</font></div></div></div></div><div>* Last updated: - <i> Mar 05, 2022, 09:46 PM India/Delhi</i></div></div></div></div></div></div></div><br><br>" \
               f"<section class='section section-l bg-white section-about' id='about'> <div class='container'> <div class='section-content'> <div class='row g-gs justify-content-between'> <div class='col-lg-7'> <div class='text-block'> <h5 class='subtitle'>About the disease</h5> <h2 class='title'>Coronavirus <br class='d-sm-none'>(<font style='color : Red;'><b>COVID-19</b></font>)</h2> <p class='lead'><strong><font style='color : Red;'><b>COVID-19</b></font> is a new illness that can affect your lungs and airways.</strong> It's caused by a virus called coronavirus. It was discovered in December 2019 in Wuhan, Hubei, China.</p><p>Common signs of infection include respiratory symptoms, fever, cough, shortness of breath and breathing difficulties. In more severe cases, infection can cause pneumonia, severe acute respiratory syndrome, kidney failure and even death.</p><p>Standard recommendations to prevent infection spread include regular hand washing, covering mouth and nose when coughing and sneezing, thoroughly cooking meat and eggs. Avoid close contact with anyone showing symptoms of respiratory illness such as coughing and sneezing.</p></div></div><div class='col-lg-5 col-xl-4'> <div class='wgs wgs-card mt-sm-2 mt-md-4 mt-lg-0 ml-lg-4 ml-xl-0'> <div class='wgs-head'> <br><br><br><h4><font style='color : blue;'>What you need to know</font></h4> </div><ul class='wgs-list'> <li>How coronavirus is spread</li><li>Symptoms of coronavirus</li><li>How to protect yourself</li><li>Treatment for coronavirus</li><li>Questions & answers</li></ul> </div></div></div></div></div></section> <br><br>" \
               f"<section class='section section-l bg-white section-symptom' id='symptoms'> <div class='container'> <div class='section-head text-center wide-lg'> <h5 class='subtitle'>What are the symptoms of <font style='color : Red;'><b>COVID-19</b></font>?</h5> <h2 class='title'>Symptoms of Coronavirus</h2><br><p>The most common symptoms of <font style='color : Red;'><b>COVID-19</b></font> are fever, tiredness, and dry cough. Some patients may have aches and pains, nasal congestion, runny nose, sore throat or diarrhea. These symptoms are usually mild and begin gradually. Also the symptoms may appear 2-14 days after exposure. </p></div><br><br><div class='section-content'> <div class='row g-gs justify-content-center'> <div class='col-sm-10 col-md-8 col-lg-4'> <div class='box2'> <div class='box2-gfx'> <img src='img/symptom-a.png' alt=''></div><div class='box2-content'><br><h5 class='title'><font style='color : red;'><b>Fever</b></font></h5> <p><strong>High Fever</strong> – this means you feel hot to touch on your chest or back (you do not need to measure your temperature). It is a common sign and also may appear in 2-10 days if you affected.</p></div></div></div><div class='col-sm-10 col-md-8 col-lg-4'> <div class='box2'> <div class='box2-gfx'> <img src='img/symptom-b.png' alt=''></div><div class='box2-content'><br><h5 class='title'><font style='color : red;'><b>Cough</b></font></h5> <p><strong>Continuous cough</strong> – this means coughing a lot for more than an hour, or 3 or more coughing episodes in 24 hours (if you usually have a cough, it may be worse than usual).</p></div></div></div><div class='col-sm-10 col-md-8 col-lg-4'> <div class='box2'> <div class='box2-gfx'> <img src='img/symptom-c.png' alt=''></div><div class='box2-content'><br><h5 class='title'><font style='color : red;'><b>Shortness of breath</b></font></h5> <p><strong>Difficulty breathing</strong> – Around 1 out of every 6 people who gets <font style='color : Red;'><b>COVID-19</b></font> becomes seriously ill and develops difficulty breathing or shortness of breath.</p></div></div></div></div></div></div></section> <br><br>" \
               f"<section class='section section-l bg-light section-Prevention' id='Prevention'> <div class='container'> <div class='section-head text-center wide-lg'> <br><h2 class='title'>Prevention</h2><br><p>The best thing you can do now is plan for how you can adapt your daily routine. Take few steps to protect yourself as Clean your hands often, Avoid close contact, Cover coughs and sneezes, clean daily used surfaces etc. The best way to prevent illness is to avoid being exposed to this virus.</p></div><br><br><div class='section-content'> <div class='row g-gs justify-content-center flex-lg-nowrap'> <div class='col-md-8 col-lg-5 col-xl-6 align-self-end'> <div class='protect-block-gfx'> <img src='img/protect.png' alt='' width='580' height='500'> </div></div><div class='col-6 col-mb-5 col-sm-6 col-lg-4 col-xl-3 flex-grow-1 order-lg-first'> <div class='protect-item negative'> <div class='protect-gfx'> <img src='img/donts-a.png' alt=''></div><div class='protect-text'> <h5 class='title'><font style='color : red;'>Avoid Close Contact</font></h5> </div></div><div class='protect-item negative'> <div class='protect-gfx'> <img src='img/donts-b.png' alt=''></div><div class='protect-text'> <h5 class='title'><font style='color : red;'>Don’t Touch Face</font></h5> </div></div><div class='protect-item negative'> <div class='protect-gfx'> <img src='img/donts-c.png' alt=''></div><div class='protect-text'> <h5 class='title'><font style='color : red;'>Social Distancing</font></h5> </div></div></div><div class='col-6 col-mb-5 col-sm-6 col-lg-4 col-xl-3 flex-grow-1 '> <div class='protect-item positive'> <div class='protect-gfx'> <img src='img/dos-a.png' alt=''></div><div class='protect-text'> <h5 class='title'><font style='color : green;'>Wash Your Hands</font></h5> </div></div><div class='protect-item positive'> <div class='protect-gfx'> <img src='img/dos-b.png' alt=''></div><div class='protect-text'> <h5 class='title'><font style='color : green;'>Drink Much Watar</font></h5> </div></div><div class='protect-item positive'> <div class='protect-gfx'> <img src='img/dos-c.png' alt=''></div><div class='protect-text'> <h5 class='title'><font style='color : green;'>Use Face Mask</font></h5> </div></div></div></div></div></div></section> <br><br>" \
               f"<section class='section section-l bg-light section-graph' id='graph'> <div class='container'> <div class='section-head text-center wide-lg'> <br><h2 class='title'>Sample of Graphical Representation</h2> </div><br><div class='col-lg-12'> <p> <h5>In the above graph we highlight, The Implementation of linear regression and polynomial regression in a given data set, </h5> </p><h5> <li> <font style='color : Red;'>Red</font> dots indicates future prediction. </li></h5> <h5> <li> <font style='color : Green;'>Green</font> line showcase the outbreak by a graphical representation. </li></h5> <h5> <li> <font style='color : Blue;'>Blue</font> dots indicates a model graph created by analyzing the data sets and predict the upcoming outbreak. </li></h5><br><h5> Prediction -{printprediction} Million Cases after {days} days Accuracy:- {printaccuracy}%</h5><br></div><br><br><canvas id='myChart' style='width:100%;'></canvas> <p> <h5><i><u>Future outcomes from the relevant dataset</u></i></h5> </p></div></section> <br><br>" \
               f"<footer class='nk-footer bg-dark tc-light has-overlay'> <br><div> <center> <font style='color : White;'> © 2022 Created & Devoloped By Team . All Rights Reserved.</font> </center> </div><br></footer></div></body><script> var xValues = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000];new Chart('myChart', [ type: 'line',  data:[ labels: xValues, datasets:[ data:[{x1}],  borderColor: 'red',fill: false]]]</script></html>" \

with open("index.html", "w") as html_file:
    html_file.write(html_content)
