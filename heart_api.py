from bottle import Bottle, request, response

from heart_model import HeartModel
from nn_model import NetworkModel


class EnableCors(object):
    def apply(self, fn, context):
        def _enable_cors(*args, **kwargs):
            # set CORS headers
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token'

            if request.method != 'OPTIONS':
                # actual request; reply with the actual response
                return fn(*args, **kwargs)

        return _enable_cors


class HeartApp:
    def __init__(self, host, port):
        self._host = host
        self._port = port
        # self.model = HeartModel()
        self.model2 = NetworkModel()
        self._app = Bottle()

        self._route()

    def _route(self):
        self._app.route('/heart_model', method="POST", callback=self.predict)

    def start(self):
        self._app.run(host=self._host, port=self._port)

    def predict(self):
        data = request.json
        result = self.model2.predict(data)
        print(result)
        return {"pred": str(result)}


if __name__ == '__main__':
    app = HeartApp(host='localhost', port=8080)
    app._app.install(EnableCors())
    app.start()
