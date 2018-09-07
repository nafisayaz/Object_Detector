
import model_dnn as dnn

dnn_model = dnn.MODEL_DNN()
model = dnn_model.create_model()

dnn_model.load_model(model)
dnn_model.fit_model(model)

