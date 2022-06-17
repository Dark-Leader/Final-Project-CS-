class Timer {

    constructor(interval, arr) {
        this.interval  = interval;
        this.counter = 0;
        this.is_running = true;
        this.arr = arr;
        this.idx = 0;
        this.timeout = null;
    }

    mainLoop() {
        if (!this.is_running) {
            return;
        }
        if (this.idx >= this.arr.length) {
            reset();
        }
        var cur = this.arr[i];
        var elapsed = 0;
        if (this.counter >= cur.timestep) {
            elapsed = playNote(cur);
            this.idx++;
        }
        this.counter += this.interval;
        this.timeout = setTimeout(this.mainLoop, this.interval - elapsed);

    }

    start() {
        this.is_running = true;
        this.mainLoop();
    }

    stop() {
        this.is_running = false;
    }

    playNote(note) {
        var now = Date.now();
        var name = note.name;

        note.currentTime = 0;
        note.play();


        var after = Date.now();
        return after - now;
    }

    reset() {
        this.counter = 0;
        this.idx = 0;
        this.is_running = false;
        if (this.timeout != null) {
            clearInterval(this.timeout);
        }
    }


}

var INTERVAL = 250; // 250ms

const keys = window.document.querySelectorAll("*[class^=\"key\"]")
console.log(keys)

const audio_file = document.getElementById('audio_file');
console.log(audio_file);

function load_notes(notes_arr) {
    var res = [];
    for(var note of notes_arr) {
        res.push(JSON.parse(note));
    }
    return res;
}

var notes = load_notes(notesArr);
console.log(notes);


var timer = new Timer(INTERVAL, notes);

console.log(timer.is_running);


function playAudio() {
    audio_file.play();
}

function stopAudio() {
    audio_file.pause();
}

function reset() {
    stopAudio();
    audio_file.currentTime = 0;
}

var play_button = window.document.getElementById('play_button');
play_button.addEventListener('click', playAudio)

var stop_button = window.document.getElementById('stop_button');
stop_button.addEventListener('click', stopAudio)

var reset_button = window.document.getElementById('reset_button');
reset_button.addEventListener('click', reset)
