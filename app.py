from flask import Flask,render_template,request
import model 
app = Flask('__name__')

valid_userid = ['kimmie', 'samantha', '00sab00', 'zburt5', 'rebecca', '1234', 'dorothy w', 'moore222', 'cassie', 'zippy', 'raeanne', 'walker557', 'joshua']
@app.route('/')
def view():
    return render_template('index.html')

@app.route('/recommend_product',methods=['POST'])
def recommend_prod():
    print(request.method)
    user_name = request.form['User Name']
    print('User name=',user_name)
    
    if  user_name in valid_userid and request.method == 'POST':
            reco_prod_20 = model.recommend_products(user_name)
            print(reco_prod_20.head())
            get_top5 = model.reco_prod_5(reco_prod_20)
            
            return render_template('index.html',column_names=get_top5.columns.values, row_data=list(get_top5.values.tolist()), zip=zip,text='Top 5 recommended products for the user '+user_name)
    elif not user_name in  valid_userid:
        return render_template('index.html',text='No Recommendation found for the user '+ user_name)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.debug=False
    app.run()
