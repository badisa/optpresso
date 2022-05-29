import React, { useState, useGlobal, setGlobal, addCallback } from "reactn";
import ReactDOM from "react-dom";

export default class Video extends React.Component {

  constructor(props) {
    super(props);
    this.state = {
      streaming: false,
    };

    this.startVideo = this.startVideo.bind(this);
  }

  componentDidMount() {
    this.startVideo();
  }

  startVideo() {
    var self = this;
    var state = this.state;
    if (state.streaming) {
      this.setState({streaming: false});
    }
    var videoWidth, videoHeight;

    var video = document.getElementById("video");
    var canvasOutput = document.getElementById("canvasOutput");
    var canvasOutputCtx = canvasOutput.getContext("2d");
    var stream = null;

    function startCamera() {
      if (state.streaming) return;
      navigator.mediaDevices
        .getUserMedia({ video: true, audio: false })
        .then(function (s) {
          stream = s;
          video.srcObject = s;
          video.play();
        })
        .catch(function (err) {
          console.log("An error occured! " + err);
          self.setState({streaming: false});
        });

      video.addEventListener(
        "canplay",
        function (ev) {
          if (!state.streaming) {
            videoWidth = video.videoWidth;
            videoHeight = video.videoHeight;
            video.setAttribute("width", videoWidth);
            video.setAttribute("height", videoHeight);
            canvasOutput.width = videoWidth;
            canvasOutput.height = videoHeight;
            self.setState({streaming: true});
            state.streaming = true;
          }
          startVideoProcessing();
        },
        false
      );

      video.addEventListener(
        "suspend",
        function (ev) {
          if (state.streaming) {
            self.setState({streaming: false});
          }
        },
        false
      );
    }

    var canvasInput = null;
    var canvasInputCtx = null;

    var canvasBuffer = null;
    var canvasBufferCtx = null;

    function startVideoProcessing() {
      if (!state.streaming) {
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
        <div class="container center-block">
          <canvas class="center-block" id="canvasOutput"></canvas>
        </div>
        {!this.state.streaming &&
          <button id="cameraReset" class="center-block" onClick={(evt) => this.startVideo()}>Reset Video</button>
        }
        <div class="invisible">
          <video id="video" class="hidden">
            Your browser does not support the video tag.
          </video>
        </div>
      </div>
    );
  }
}