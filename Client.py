import socketio

sio = socketio.Client()

@sio.on('download')
def on_message(data):
    print('I received a message!')

if __name__ == "__main__":

    sio.connect('http://143.248.6.25:35005', transports=['websocket'])

    import time
    time.sleep(1)

    sio.emit('my event2')
    print(sio.connected)
    for i in range(0, 1000, 3):
        data = []
        data.append('TUM')
        data.append('TUM2_desk_70.color')
        data.append(i)
        sio.emit('my event2')