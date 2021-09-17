import React, { useState, useGlobal, setGlobal } from "reactn";
import _ from 'lodash';
import Video from "../src/video";
import {median, mean, std} from "../src/math";

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
  xmlHttp.open("POST", `http://localhost:8000${url}`, true); // true for asynchronous
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
    global.window.history.pushState({}, "", "/predict/");
     const navEvent = new PopStateEvent('popstate');
    global.window.dispatchEvent(navEvent);
  }
  setCapture() {
    event.preventDefault();
    this.props.setMode("capture");
    global.window.history.pushState({}, "", "/capture/");
    const navEvent = new PopStateEvent('popstate');
    global.window.dispatchEvent(navEvent);
  }
  
  render() {
    const isPredict = false;
    return (
      <nav className="navbar navbar-expand-sm navbar-dark bg-dark fixed-top">
        <div className="container-fluid">
          <a className="navbar-brand" href="#">
            Optpresso
          </a>
          <button
            className="navbar-toggler"
            type="button"
            aria-expanded="false"
            aria-label="Toggle navigation"
          >
            <span className="navbar-toggler-icon"></span>
          </button>

          <div className="collapse navbar-collapse" id="navbarsExampleDefault">
            <ul className="navbar-nav me-auto mb-1 mb-sm-0">
              <li className="nav-item">
                <a
                  className={`nav-link ${
                    !isPredict ? "active" : ""
                  }`}
                  href="#"
                  onClick={this.setCapture}
                >
                  Capture
                </a>
              </li>
              <li className="nav-item">
                <a
                  className={`nav-link ${
                    isPredict ? "active" : ""
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
      <div className="container">
        <div className="row">
          <div className="col-md-5 offset-1 config">
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
          <div className="col-md-5 offset-1 config">
            <button onClick={this.handlePredict}>Predict</button>
          </div>
        </div>
      </div>
    );
  }
}

const CaptureControl2 = () => {
  const [config, set_config] = React.useGlobal("config");
  const [pullData, set_pullData] = React.useGlobal("pullData");
  const [buttonEnabled, set_buttonEnabled] = React.useState(true);
  
  const persistConfig = () => {
    let form = new FormData();
    _.forEach(config, (v, k) => {
      form.set(k, v || "")
    });
    httpPostAsync(
      "/config/",
      form,
      () => {
    });
  };
  
  const handleSubmit = () => {
    set_buttonEnabled(false);
    captureImage(
      { ...config, ...pullData },
      (data) => {
        set_buttonEnabled(true);
      }
    );
  }
  
  return (
    <div className="container">
      <div className="row">
        <div className="col-md-5 offset-1 config">
          <h3>Configuration</h3>
          <button type="button" onClick={persistConfig}>
            Update Config
          </button>
          <hr></hr>
          {_.map(config, (value, key) => {
            const labelName = key.replace(" ", "%nbsp;") ? key : "";
            return (
              <div key={key} className="form-group">
                <label>
                  {labelName}:
                  <input
                    type="text"
                    name={key}
                    value={value}
                    onChange={
                      (e) => set_config({
                        ...config,
                        [key]: e.target.value,
                      })
                    }
                  />
                </label>
              </div>
            );
          })}
        </div>
        <div className="col-md-5 offset-1 config">
          <h3>Capture</h3>
          <button
            type={"button"}
            disabled={!buttonEnabled}
            onClick={(evt) => handleSubmit(evt)}
          >
            Capture
          </button>
          <hr></hr>
          {_.map(pullData, (value, key) => {
            const labelName = key.replace(" ", "%nbsp;") ? key : "";
            const isBoolean = typeof value === "boolean";
            const extraProps = {};
            if (isBoolean) {
              extraProps.checked = value;
            } else {
              extraProps.value = value;
            }
            return (
              <div key={key} className="form-group">
                <label>
                  {labelName}:
                  <input
                    type={isBoolean ? "checkbox" : "text"}
                    name={key}
                    onChange={(e) => set_pullData({
                      ...pullData,
                      [key]: (isBoolean ? e.target.checked : e.target.value),
                    })}
                    {...extraProps}
                  />
                </label>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
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

  // componentDidUpdate(oldProps) {
  //   if (oldProps !== this.props) {
  //     let newState = {
  //       config: { ...this.props.config },
  //       pullData: this.state.pullData,
  //     };
  //     this.setState(newState);
  //   }
  // }

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
      <div className="container">
        <div className="row">
          <div className="col-md-5 offset-1 config">
            <h3>Configuration</h3>
            <button onClick={(evt) => this.handleConfigUpdate(evt)}>
              Update Config
            </button>
            <hr></hr>
            {Object.entries(this.state.config).map(([key, value]) => {
              const labelName = key.replace(" ", "%nbsp;") ? key : "";
              return (
                <div key={key} className="form-group">
                  <label>
                    {labelName}:
                    <input
                      type="text"
                      name={key}
                      value={value}
                      onChange={
                        (e) => set_config({
                          ...config,
                          key: e.target.value,
                        })}
                    />
                  </label>
                </div>
              );
            })}
          </div>
          <div className="col-md-5 offset-1 config">
            <h3>Capture</h3>
            <button onClick={(evt) => this.handleSubmit(evt)}>Capture</button>
            <hr></hr>
            {Object.entries(this.state.pullData).map(([key, value]) => {
              const labelName = key.replace(" ", "%nbsp;") ? key : "";
              if (typeof value === "boolean") {
                return (
                  <div key={key} className="checkbox">
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
                  <div key={key} className="form-group">
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


function BaseView() {
  const [mode, setMode] = useGlobal("mode");
  const [config, setConfig] = useGlobal("config");
  const [pullData, setPullData] = useGlobal("pullData");

  const [currentPath, setCurrentPath] = useState();

  const getExistingConfig = () => {
    // httpGetAsync("/config/", function (data) {
    //   const savedConfig = JSON.parse(data);
    // });
    console.log('wtf')
    setConfig({
      machine: 'somestring',
      grinder: 'anotherstring',
    });
  };
  React.useEffect(() => {
    getExistingConfig();
    
    const onLocationChange = () => {
      // update path state to current window URL
      setCurrentPath(global.window.location.pathname);
    }

    // listen for popstate event
    global.window.addEventListener('popstate', onLocationChange);
    
    return () => {
      global.window.removeEventListener('popstate', onLocationChange);
    }
  }, []);
  
  // let controls;
  // if (global.window.location.pathname == "/predict/") {
  //   controls = <PredictControl config={config} setConfig={setConfig} />;
  // } else {
  //   controls = ;
  // }
  return (
    <main>
      <Nav mode={mode} setMode={setMode}></Nav>
      <Video />
      <CaptureControl2 />
    </main>
  );
}

export default BaseView;
