/*
 * ATTENTION: The "eval" devtool has been used (maybe by default in mode: "development").
 * This devtool is neither made for production nor for readable output files.
 * It uses "eval()" calls to create a separate source file in the browser devtools.
 * If you are trying to read the output file, select a different devtool (https://webpack.js.org/configuration/devtool/)
 * or disable the default devtool with "devtool: false".
 * If you are looking for production-ready output files, see mode: "production" (https://webpack.js.org/configuration/mode/).
 */
/******/ (() => { // webpackBootstrap
/******/ 	var __webpack_modules__ = ({

/***/ "./tgpu.js":
/*!*****************!*\
  !*** ./tgpu.js ***!
  \*****************/
/***/ (() => {

eval("async function test() {\n  console.log(\"Michael Nath\");\n  \n  if (!navigator.gpu) {\n    console.log(\"WebGPU is not supported on this browser.\");\n    document.body.innerHTML = \"<p>WebGPU is not supported on this browser.</p>\";\n    return;\n  }\n\n  console.log(navigator.gpu);\n  document.body.innerHTML = `<pre>${JSON.stringify(navigator.gpu, null, 2)}</pre>`;\n  \n  const adapter = await navigator.gpu.requestAdapter();\n  console.log(adapter);\n  if (!adapter) {\n    document.body.innerHTML += \"<p>Failed to get GPU adapter.</p>\";\n    return;\n  }\n  \n  const device = await adapter.requestDevice();\n  console.log(device.adapterInfo);\n  \n  document.body.innerHTML += `<pre>${adapter.name}</pre>`;\n}\n\n// Run when the page is fully loaded\ndocument.addEventListener(\"DOMContentLoaded\", test);\n\n\n//# sourceURL=webpack:///./tgpu.js?");

/***/ })

/******/ 	});
/************************************************************************/
/******/ 	
/******/ 	// startup
/******/ 	// Load entry module and return exports
/******/ 	// This entry module can't be inlined because the eval devtool is used.
/******/ 	var __webpack_exports__ = {};
/******/ 	__webpack_modules__["./tgpu.js"]();
/******/ 	
/******/ })()
;