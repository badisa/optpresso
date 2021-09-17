import React, { useState, useGlobal, setGlobal, addCallback } from "reactn";
import ReactDOM from "react-dom";

export default class Video extends React.Component {
  componentDidMount() {
    var videoWidth, videoHeight;

    // whether streaming video from the camera.
    var streaming = false;

    var video = document.getElementById("video");
    var canvasOutput = document.getElementById("canvasOutput");
    var canvasOutputCtx = canvasOutput.getContext("2d");
    var stream = null;

    function startCamera() {
      if (streaming) return;
      navigator.mediaDevices
        .getUserMedia({ video: true, audio: false })
        .then(function (s) {
          stream = s;
          video.srcObject = s;
          video.play();
        })
        .catch(function (err) {
          console.log("An error occured! " + err);
        });

      video.addEventListener(
        "canplay",
        function (ev) {
          if (!streaming) {
            videoWidth = video.videoWidth;
            videoHeight = video.videoHeight;
            video.setAttribute("width", videoWidth);
            video.setAttribute("height", videoHeight);
            canvasOutput.width = videoWidth;
            canvasOutput.height = videoHeight;
            streaming = true;
          }
          startVideoProcessing();
        },
        false
      );
    }

    var canvasInput = null;
    var canvasInputCtx = null;

    var canvasBuffer = null;
    var canvasBufferCtx = null;

    function startVideoProcessing() {
      if (!streaming) {
        console.warn("Please startup your webcam");
        return;
      }
      canvasInput = document.createElement("canvas");
      canvasInput.width = videoWidth;
      canvasInput.height = videoHeight;
      canvasInputCtx = canvasInput.getContext("2d");

      canvasBuffer = document.createElement("canvas");
      canvasBuffer.width = videoWidth;
      canvasBuffer.height = videoHeight;
      canvasBufferCtx = canvasBuffer.getContext("2d");

      requestAnimationFrame(processVideo);
    }

    function processVideo() {
      canvasInputCtx.drawImage(video, 0, 0, videoWidth, videoHeight);
      var imageData = canvasInputCtx.getImageData(
        0,
        0,
        videoWidth,
        videoHeight
      );
      canvasOutputCtx.drawImage(canvasInput, 0, 0, videoWidth, videoHeight);
      requestAnimationFrame(processVideo);
    }

    startCamera();
  }

  render() {
    return (
      <div>
        <div className="container">
          <canvas className="center-block" id="canvasOutput"></canvas>
        </div>
        <div className="invisible">
          <video id="video" className="hidden">
            Your browser does not support the video tag.
          </video>
        </div>
      </div>
    );
  }
}
