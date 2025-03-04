const path = require('path');

module.exports = {
  mode: 'development',
  entry: './tgpu.js',
  output: {
    filename: 'bundle.js',
    path: path.resolve(__dirname, 'dist'),
  },
  resolve: {
    alias: {
      'node:events': 'events',
      // Add any other node: prefixed modules that cause problems
    }
  }
};