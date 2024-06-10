require('dotenv').config();
 
const Hapi = require('@hapi/hapi');
const routes = require('../server/routes');
const loadModel = require('../services/loadModel');
const InputError = require('../exceptions/InputError');

(async () => {
    const server = Hapi.server({
        port: 3000,
        host: '34.101.103.62',
        routes: {
            cors: {
              origin: ['*'],
            },
        },
    });
 
    const model = await loadModel();
    if (model === null) {
        console.log('Failed to load model');
    } else {
        console.log('Model loaded successfully');
    }
    server.app.model = model;
 
    server.route(routes);
 
    server.ext('onPreResponse', function (request, h) {
        const response = request.response;
 
        // if (response instanceof InputError) {
        //     const newResponse = h.response({
        //         status: 'fail',
        //         message: `${response.message} Silakan gunakan foto lain.`
        //     })
        //     newResponse.code(response.statusCode)
        //     return newResponse;
        // }
        if (response.isBoom && response.output.statusCode === 413) {
            const newResponse = h.response({
                status: 'fail',
                message: 'Payload content length greater than maximum allowed: 1000000',
            });
            
            newResponse.code(413);
            return newResponse;
        }
        
        if (response instanceof InputError || response.isBoom) {
            const statusCode = response instanceof InputError ? response.statusCode : response.output.statusCode;
            const newResponse = h.response({
                status: 'fail',
                message: 'Terjadi kesalahan dalam melakukan prediksi',
            });

            newResponse.code(parseInt(statusCode));
            return newResponse;
        }
 
        return h.continue;
    });
 
    await server.start();
    console.log(`Server start at: ${server.info.uri}`);
})();