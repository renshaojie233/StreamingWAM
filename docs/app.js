const RELEASE_BASE = location.hostname.endsWith("github.io")
  ? "videos"
  : "https://github.com/renshaojie233/StreamingWAM/releases/download/supplementary-v1";

const assetUrl = name => `${RELEASE_BASE}/${name}`;
const pad = value => String(value).padStart(2, "0");

const TASKS = [
  ["01", "Conveyor apple to bowl", "Dynamic"],
  ["02", "Moving-platform apple to bowl", "Dynamic"],
  ["03", "Whack-a-mole", "Dynamic"],
  ["04", "Open drawer", "Static"],
  ["05", "Close drawer", "Static"],
  ["06", "Weigh apple, then place into bowl", "Static"],
  ["07", "Stack blocks", "Static"],
];

const TASK_SLUGS = {
  "01": "conveyor-apple-to-bowl",
  "02": "moving-platform-apple-to-bowl",
  "03": "hit-the-mole",
  "04": "open-drawer",
  "05": "close-drawer",
  "06": "weigh-apple-then-bowl",
  "07": "stack-blocks",
};

document.querySelectorAll("[data-release-asset]").forEach(video => {
  video.src = assetUrl(video.dataset.releaseAsset);
});

const ROLLOUT_GROUPS = [
  { container: "static-task-grid", tasks: ["07", "04", "06"] },
  { container: "dynamic-task-grid", tasks: ["01", "02", "03"] },
];

ROLLOUT_GROUPS.forEach(({ container, tasks }) => {
  const taskGrid = document.getElementById(container);
  tasks.forEach(number => {
    const [, title, group] = TASKS.find(([taskNumber]) => taskNumber === number);
    const filename = `01-task-demonstrations__task-${number}-${TASK_SLUGS[number]}.mp4`;
    const version = number === "01" ? "?v=b562aba0" : "";
    const article = document.createElement("article");
    article.className = "task-card";
    article.innerHTML = `
      <video controls muted playsinline preload="metadata"
        poster="assets/posters/${filename.replace(".mp4", ".jpg")}${version}" src="${assetUrl(filename)}${version}"></video>
      <div class="task-meta"><strong>${title}</strong><span>${number} · ${group}</span></div>`;
    taskGrid.appendChild(article);
  });
});

const featuredMain = document.getElementById("featured-rollout-main");
const featuredInputs = [...document.querySelectorAll(".featured-input-video")];

function syncFeaturedInputs(play = false) {
  const time = featuredMain.currentTime;
  featuredInputs.forEach(video => {
    if (Number.isFinite(video.duration) && Math.abs(video.currentTime - time) > .15) {
      video.currentTime = Math.min(time, video.duration);
    }
  });
  if (play) Promise.allSettled(featuredInputs.map(video => video.play()));
}

featuredMain.addEventListener("play", () => syncFeaturedInputs(true));
featuredMain.addEventListener("pause", () => featuredInputs.forEach(video => video.pause()));
featuredMain.addEventListener("seeking", () => syncFeaturedInputs(false));
featuredMain.addEventListener("timeupdate", () => syncFeaturedInputs(false));
featuredMain.addEventListener("ended", () => featuredInputs.forEach(video => video.pause()));

const COMPARISON_TASKS = {
  "01": "Put apple into bowl",
  "02": "Weigh apple, then put into bowl",
};
const METHODS = [
  { label: "StreamingWAM", model: "model-b", className: "ours", note: "Ours" },
  { label: "π0.5", model: "model-a", className: "", note: "Baseline" },
  { label: "FastWAM-Joint", model: "model-c", className: "", note: "Baseline" },
];
const state = { trial: "01", camera: "04" };
const comparisonVideos = [];

function makeButtons(containerId, values, labeler, key) {
  const container = document.getElementById(containerId);
  values.forEach(value => {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.value = value;
    button.textContent = labeler(value);
    button.addEventListener("click", () => {
      state[key] = value;
      updateComparison();
    });
    container.appendChild(button);
  });
}

makeButtons("trial-controls", ["01", "02"], value => `Trial ${Number(value)}`, "trial");
makeButtons("camera-controls", ["01", "02", "03", "04"], value => value === "04" ? "Recorded view" : `Input ${Number(value)}`, "camera");

const comparisonGroups = document.getElementById("comparison-groups");
Object.entries(COMPARISON_TASKS).forEach(([task, title]) => {
  const section = document.createElement("section");
  section.className = "comparison-group";
  section.innerHTML = `
    <h3 class="comparison-title" data-comparison-title="${task}"></h3>
    <div class="comparison-grid"></div>`;
  const grid = section.querySelector(".comparison-grid");
  METHODS.forEach(method => {
    const article = document.createElement("article");
    article.className = `comparison-card ${method.className}`;
    article.innerHTML = `
      <div class="comparison-label"><strong>${method.label}</strong><span>${method.note}</span></div>
      <video controls muted playsinline preload="auto"></video>`;
    grid.appendChild(article);
    comparisonVideos.push({ video: article.querySelector("video"), method, task });
  });
  comparisonGroups.appendChild(section);
});

const timeline = document.getElementById("timeline");
const readout = document.getElementById("time-readout");
let duration = 0;
let dragging = false;

function comparisonFilename(task, method) {
  return `02-blind-model-comparison__task-${task}__${method.model}__trial-${state.trial}__camera-${state.camera}.mp4`;
}

function posterFilename(task, method) {
  return `assets/posters/02-blind-model-comparison__task-${task}__${method.model}__trial-${state.trial}__camera-04.jpg`;
}

function setActiveButtons(containerId, value) {
  document.querySelectorAll(`#${containerId} button`).forEach(button => {
    button.classList.toggle("active", button.dataset.value === value);
  });
}

function formatTime(seconds) {
  if (!Number.isFinite(seconds)) return "00:00.0";
  const minutes = Math.floor(seconds / 60);
  const remaining = seconds - minutes * 60;
  return `${pad(minutes)}:${remaining.toFixed(1).padStart(4, "0")}`;
}

function updateDuration() {
  const durations = comparisonVideos.map(item => item.video.duration).filter(Number.isFinite);
  if (durations.length === comparisonVideos.length) {
    duration = Math.min(...durations);
    timeline.max = duration;
    readout.value = `${formatTime(0)} / ${formatTime(duration)}`;
  }
}

let comparisonLoadVersion = 0;

function updateComparison() {
  const loadVersion = ++comparisonLoadVersion;
  comparisonVideos.forEach(({ video, method, task }) => {
    video.pause();
    const filename = comparisonFilename(task, method);
    const version = filename === "02-blind-model-comparison__task-02__model-b__trial-01__camera-04.mp4"
      ? "?v=42cc6b6e"
      : "";
    video.style.setProperty("--comparison-aspect", state.camera === "04" ? "1280 / 904" : "16 / 9");
    video.addEventListener("loadedmetadata", () => {
      if (loadVersion !== comparisonLoadVersion) return;
      video.currentTime = Math.min(0.01, video.duration || 0.01);
    }, { once: true });
    video.src = `${assetUrl(filename)}${version}`;
    if (state.camera === "04") {
      video.poster = posterFilename(task, method);
    } else {
      video.removeAttribute("poster");
    }
    video.load();
  });
  duration = 0;
  timeline.max = 1;
  timeline.value = 0;
  readout.value = "00:00.0 / 00:00.0";
  document.querySelectorAll("[data-comparison-title]").forEach(title => {
    title.textContent = `${COMPARISON_TASKS[title.dataset.comparisonTitle]} · Trial ${Number(state.trial)}`;
  });
  setActiveButtons("trial-controls", state.trial);
  setActiveButtons("camera-controls", state.camera);
}

comparisonVideos.forEach(({ video }) => video.addEventListener("loadedmetadata", updateDuration));

document.getElementById("play-all").addEventListener("click", async () => {
  const time = Math.min(Number(timeline.value), duration || Infinity);
  comparisonVideos.forEach(({ video }) => { video.currentTime = time; });
  await Promise.allSettled(comparisonVideos.map(({ video }) => video.play()));
});
document.getElementById("pause-all").addEventListener("click", () => comparisonVideos.forEach(({ video }) => video.pause()));
document.getElementById("restart-all").addEventListener("click", () => {
  comparisonVideos.forEach(({ video }) => { video.pause(); video.currentTime = 0; });
  timeline.value = 0;
  readout.value = `${formatTime(0)} / ${formatTime(duration)}`;
});
timeline.addEventListener("pointerdown", () => { dragging = true; });
timeline.addEventListener("pointerup", () => { dragging = false; });
timeline.addEventListener("input", () => {
  const time = Number(timeline.value);
  comparisonVideos.forEach(({ video }) => { video.currentTime = time; });
  readout.value = `${formatTime(time)} / ${formatTime(duration)}`;
});
comparisonVideos[0].video.addEventListener("timeupdate", () => {
  const time = comparisonVideos[0].video.currentTime;
  if (!dragging) timeline.value = Math.min(time, duration || time);
  comparisonVideos.slice(1).forEach(({ video }) => {
    if (!video.paused && Math.abs(video.currentTime - time) > .18) video.currentTime = time;
  });
  readout.value = `${formatTime(time)} / ${formatTime(duration)}`;
});
updateComparison();
