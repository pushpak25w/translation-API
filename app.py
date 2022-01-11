from flask import Flask
import flask.scaffold
flask.helpers._endpoint_from_view_func = flask.scaffold._endpoint_from_view_func
from flask_restful import Api,Resource, reqparse
from model import LSTM_model
app= Flask(__name__)
api=Api(app)
args=reqparse.RequestParser()
args.add_argument("lang",type=str,help="send language to be translated",required=True)
args.add_argument("trans",type=str,help="send language to which you want to translate your sentence",required=True)
args.add_argument("sentence",type=str,help="send sentence to be translated",required=True)


mar=LSTM_model('./files/mar/eng_words.pickle','./files/mar/mar_words.pickle','./files/mar/eng_to_mar.h5',20,20,256)
fra=LSTM_model('./files/fra/eng_words.pickle','./files/fra/fra_words.pickle','./files/fra/eng_to_french.h5',20,20,60)
#hindi=LSTM_model('./files/hindi/eng_words.pickle','./files/hindi/hindi_words.pickle','./files/hindi/eng_to_hindi.h5',20,20,300)

class Category(Resource):
    def put(self,sentence):
        tmpArgs=args.parse_args()
        inp=tmpArgs['sentence']
        out=" "
        if tmpArgs['trans']=='marathi':
            out=mar.shape_data(inp)
            out=mar.pad_data(out)
            out=mar.decode_sequence(out)[:-4]
        elif tmpArgs['trans']=='hindi' :
            out=hindi.shape_data(inp)
            out=hindi.pad_data(out)
            out=hindi.decode_sequence(out)[:-4]
        elif tmpArgs['trans']=='fra' :
            out=fra.shape_data(inp)
            out=fra.pad_data(out)
            out=fra.decode_sequence(out)[:-4]
        return {'Input English sentence':inp,'Predicted Translation':out}


api.add_resource(Category,"/category/<string:sentence>")

if __name__=="__main__":
    app.run(debug=True)
