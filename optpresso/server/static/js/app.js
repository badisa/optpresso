import React, { useState, useGlobal, setGlobal, addCallback } from "reactn";
import ReactDOM from "react-dom";

import "../sass/style.scss";
import "bootstrap/dist/css/bootstrap.min.css";

import Video from "./video.js";
import {median, mean, std} from "./math.js";

setGlobal({
  config: {},
  pullData: {
    pullTime: 30,
    grindSetting: "",
    coffee: "",
    gramsIn: 18,
    gramsOut: 36,
  },
});

function predictPullTime(callback) {
  let form = new FormData();
  postImage(form, "/predict/", callback);
}

function captureImage(data, callback) {
  let form = new FormData();
  for (const [key, value] of Object.entries(data)) {
    form.append(key, value);
  }
  postImage(form, "/capture/", callback);
}

function postImage(formData, url, callback) {
  let canvasOutput = document.getElementById("canvasOutput");
  formData.append("image", canvasOutput.toDataURL());
  httpPostAsync(url, formData, callback);
}

function httpPostAsync(url, formData, callback) {
  var xmlHttp = new XMLHttpRequest();
  xmlHttp.onreadystatechange = function () {
    if (xmlHttp.readyState == 4 && xmlHttp.status < 300)
      callback(xmlHttp.responseText);
  };
  xmlHttp.open("POST", url, true); // true for asynchronous
  xmlHttp.send(formData);
}

function httpGetAsync(url, callback) {
  var xmlHttp = new XMLHttpRequest();
  xmlHttp.onreadystatechange = function () {
    if (xmlHttp.readyState == 4 && xmlHttp.status < 300)
      callback(xmlHttp.responseText);
  };
  xmlHttp.open("GET", url, true); // true for asynchronous
  xmlHttp.send();
}

class Nav extends React.Component {
  constructor(props) {
    super(props);
    this.setPredict = this.setPredict.bind(this);
    this.setCapture = this.setCapture.bind(this);
  }
  setPredict(event) {
    event.preventDefault();
    this.props.setMode("predict");
    window.history.pushState({}, "", "/predict/");
     const navEvent = new PopStateEvent('popstate');
    window.dispatchEvent(navEvent);
  }
  setCapture() {
    event.preventDefault();
    this.props.setMode("capture");
    window.history.pushState({}, "", "/capture/");
    const navEvent = new PopStateEvent('popstate');
    window.dispatchEvent(navEvent);
  }
  render() {
    return (
      <nav class="navbar navbar-expand-sm navbar-dark bg-dark fixed-top">
        <div class="container-fluid">
          <a class="navbar-brand" href="#">
            Optpresso
          </a>
          <button
            class="navbar-toggler"
            type="button"
            aria-expanded="false"
            aria-label="Toggle navigation"
          >
            <span class="navbar-toggler-icon"></span>
          </button>

          <div class="collapse navbar-collapse" id="navbarsExampleDefault">
            <ul class="navbar-nav me-auto mb-1 mb-sm-0">
              <li class="nav-item">
                <a
                  className={`nav-link ${
                    window.location.pathname != "/predict/" ? "active" : ""
                  }`}
                  href="#"
                  onClick={this.setCapture}
                >
                  Capture
                </a>
              </li>
              <li class="nav-item">
                <a
                  className={`nav-link ${
                    window.location.pathname == "/predict/" ? "active" : ""
                  }`}
                  href="#"
                  onClick={this.setPredict}
                >
                  Predict
                </a>
              </li>
            </ul>
          </div>
        </div>
      </nav>
    );
  }
}

class PredictControl extends React.Component {
  constructor(props) {
    super(props);
    this.state = { predictions: [] };

    this.handlePredict = this.handlePredict.bind(this);
    this.clearPredictions = this.clearPredictions.bind(this);
  }

  handlePredict(event) {
    event.target.disabled = true;
    let self = this;
    predictPullTime(function (data) {
      const obj = JSON.parse(data);
      self.state.predictions.push(obj.prediction);
      self.setState({ ...self.state });
      event.target.disabled = false;
    });
  }

  clearPredictions(event) {
    this.setState({ predictions: [] });
  }

  render() {
    const listItems = this.state.predictions.map((pred) => <li>{pred}</li>);
    let avg = 0.0;
    let std_val = 0.0;
    let med = 0.0;
    if (this.state.predictions.length > 1) {
      avg = mean(this.state.predictions);
      med = median(this.state.predictions);
      std_val = std(this.state.predictions);
    }
    avg = Number(avg.toFixed(2));
    std_val = Number(std_val.toFixed(2));
    med = Number(med.toFixed(2));
    return (
      <div class="container">
        <div class="row">
          <div class="col-md-5 offset-1 config">
            <h3>Predictions</h3>
            <button onClick={this.clearPredictions}>Clear</button>
            <hr></hr>
            Median:{med}
            <br></br>
            Mean:{avg}
            <br></br>
            Std: {std_val}
            <ul>{listItems}</ul>
          </div>
          <div class="col-md-5 offset-1 config">
            <button onClick={this.handlePredict}>Predict</button>
          </div>
        </div>
      </div>
    );
  }
}

class CaptureControl extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      config: this.props.config,
      pullData: this.props.pullData,
    };

    this.handleChange = this.handleChange.bind(this);
    this.handleSubmit = this.handleSubmit.bind(this);
    this.handleConfigUpdate = this.handleConfigUpdate.bind(this);
  }

  componentDidUpdate(oldProps) {
    if (oldProps !== this.props) {
      let newState = {
        config: { ...this.props.config },
        pullData: this.state.pullData,
      };
      this.setState(newState);
    }
  }

  handleConfigUpdate(event) {
    event.target.disabled = true;
    let form = new FormData();
    for (let [key, value] of Object.entries(this.state.config)) {
      if (value == null) {
        value = "";
      }
      form.append(key, value);
    }
    httpPostAsync("/config/", form, () => {
      event.target.disabled = false;
    });
  }

  handleChange(event) {
    let val = event.target.value;
    if (event.target.type == "checkbox") {
      val = event.target.checked;
    }
    if (this.state.config.hasOwnProperty(event.target.name)) {
      this.setState({
        config: {
          ...this.state.config,
          [event.target.name]: val,
        },
      });
    } else {
      this.setState({
        pullData: {
          ...this.state.pullData,
          [event.target.name]: val,
        },
      });
      this.props.setPullData({...this.state.pullData, [event.target.name]: val});
    }
  }

  handleSubmit(event) {
    event.target.disabled = true;
    this.props.setConfig(this.state.config);
    event.preventDefault();
    captureImage({ ...this.state.config, ...this.state.pullData }, (data) => {
      event.target.disabled = false;
    });
  }

  render() {
    return (
      <div class="container">
        <div class="row">
          <div class="col-md-5 offset-1 config">
            <h3>Configuration</h3>
            <button onClick={(evt) => this.handleConfigUpdate(evt)}>
              Update Config
            </button>
            <hr></hr>
            {Object.entries(this.state.config).map(([key, value]) => {
              const labelName = key.replace(" ", "%nbsp;") ? key : "";
              return (
                <div class="form-group">
                  <label>
                    {labelName}:
                    <input
                      type="text"
                      name={key}
                      value={value}
                      onChange={(evt) => this.handleChange(evt)}
                    />
                  </label>
                </div>
              );
            })}
          </div>
          <div class="col-md-5 offset-1 config">
            <h3>Capture</h3>
            <button onClick={(evt) => this.handleSubmit(evt)}>Capture</button>
            <hr></hr>
            {Object.entries(this.state.pullData).map(([key, value]) => {
              const labelName = key.replace(" ", "%nbsp;") ? key : "";
              if (typeof value === "boolean") {
                return (
                  <div class="checkbox">
                    <label>
                      {labelName}:
                      <input
                        name={key}
                        type="checkbox"
                        checked={value}
                        onClick={(evt) => this.handleChange(evt)}
                      />
                    </label>
                  </div>
                );
              } else {
                return (
                  <div class="form-group">
                    <label>
                      {labelName}:
                      <input
                        type="text"
                        name={key}
                        value={value}
                        onChange={(evt) => this.handleChange(evt)}
                      />
                    </label>
                  </div>
                );
              }
            })}
          </div>
        </div>
      </div>
    );
  }
}


function CaptureComponent() {
  const [mode, setMode] = useGlobal("mode");
  const [config, setConfig] = useGlobal("config");
  const [pullData, setPullData] = useGlobal("pullData");

  const [currentPath, setCurrentPath] = useState(window.location.pathname);

  const getExistingConfig = () => {
    httpGetAsync("/config/", function (data) {
      const savedConfig = JSON.parse(data);
      setConfig(savedConfig);
    });
  };
  React.useEffect(() => {
    getExistingConfig();
    const onLocationChange = () => {
      // update path state to current window URL
      setCurrentPath(window.location.pathname);
    }

    // listen for popstate event
    window.addEventListener('popstate', onLocationChange);
  }, []);
  let controls;
  if (window.location.pathname == "/predict/") {
    controls = <PredictControl config={config} setConfig={setConfig} />;
  } else {
    controls = <CaptureControl config={config} setConfig={setConfig} pullData={pullData} setPullData={setPullData}/>;
  }
  return (
    <main>
      <Nav mode={mode} setMode={setMode}></Nav>
      <Video />
      {controls}
    </main>
  );
}

ReactDOM.render(<CaptureComponent />, document.getElementById("root"));
