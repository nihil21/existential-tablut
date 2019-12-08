---------- Existential_Tablut ----------
This project is structured as follows:
- The 'src' folder contains all the scripts implementing the neuroevolution which allowed us to produce the final NNs.
- The 'test' folder contains some tests.
- Finally, the 'exec' folder contains the code of the client; in particular, to execute the client one must run the 'runmyplayer' script, giving as arguments the role ('White' or 'Black'), the timeout and the server ip:

    ./runmyplayer <role> <timeout> <server_ip>
    
The external libraries needed to run the scripts are:
- keras
- theano
- sklearn
Please note that keras relies by default on the tensorflow backend, thus it must explicitly told to use theano instead (by modifying ~/.keras/keras.json).
Our vm in which the client will run is already configured to do so.

Credits:
Lorenzo Mario Amorosa (https://github.com/Lostefra)
Mattia Orlandi (https://github.com/nihil21)
Giacomo Pinardi (https://github.com/GiacomoPinardi)
Giorgio Renzi (https://github.com/gioggio)