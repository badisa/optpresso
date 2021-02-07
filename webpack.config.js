const path = require('path');

module.exports = {
  entry: './optpresso/server/static/js/app.js',
  target: 'web',
  optimization: {
    minimize: false
  },
  output: {
    filename: 'optpresso.js',
    path: path.resolve(__dirname, 'optpresso/server/static/compiled/'),
    libraryTarget: 'window'
  },
  module: {
    rules: [
      {
        test: /\.m?js$/,
        exclude: /(node_modules|bower_components)/,
        use: {
          loader: 'babel-loader',
          options: {
            presets: ['@babel/preset-react']
          }
        }
      },
      {
        test: /\.(scss|css)$/,
        use: ['style-loader', 'css-loader', 'sass-loader'],
      },
    ]
  },
};
