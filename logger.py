import logging

# logger instance
logger = logging.getLogger('root')

# formatter
formatter = logging.Formatter("[%(asctime)s %(filename)s:%(lineno)s - %(funcName)s()] %(message)s")

# handler (stream, file)
streamHandler = logging.StreamHandler()
fileHandler = logging.FileHandler('log/server.log')

# set formatter on logger instance
streamHandler.setFormatter(formatter)
fileHandler.setFormatter(formatter)

# set handler on logger instance
logger.addHandler(streamHandler)
logger.addHandler(fileHandler)

logger.setLevel(logging.INFO)
logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib').disabled = True
