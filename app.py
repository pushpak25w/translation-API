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


mar=LSTM_model('./files/mar/eng_words.pickle','./files/mar/mar_words.pickle','./files/mar/eng_to_mar.h5',20,20)

class Category(Resource):
    def put(self,sentence):
        tmpArgs=args.parse_args()
        inp=tmpArgs['sentence']
        out=" "
        if tmpArgs['trans']=='marathi':
            out=mar.shape_data(inp)
            out=mar.pad_data(out)
            out=mar.decode_sequence(out)[:-4]
        return {'Input English sentence':inp,'Predicted marathi Translation':out}



api.add_resource(Category,"/category/<string:sentence>")

if __name__=="__main__":
    app.run(debug=True)
