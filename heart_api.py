from bottle import Bottle, request, response

from heart_model import HeartModel


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
        return {"pred": str(result[0])}


if __name__ == '__main__':
    app = HeartApp(host='localhost', port=8080)
    app.start()
