from bottle import Bottle, request, response

from heart_model import HeartModel


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
        self.model = HeartModel()
        self._app = Bottle()

        self._route()

    def _route(self):
        self._app.route('/heart_model', method="POST", callback=self.predict)

    def start(self):
        self._app.run(host=self._host, port=self._port)

    def predict(self):
        data = request.json
        result = self.model.predict(data)
        print(str(result[0]))

        response.content_type = 'application/json'
        response.headers['Access-Control-Allow-Origin'] = '*'
        return {"pred": str(result[0])}


if __name__ == '__main__':
    app = HeartApp(host='localhost', port=8080)
    app._app.install(EnableCors())
    app.start()
