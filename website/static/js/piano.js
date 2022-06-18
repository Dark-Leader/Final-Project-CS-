const INTERVAL = 250; // 250ms

const keys = window.document.querySelectorAll('.key');
console.log(keys)

const audio_file = document.getElementById('audio_file');
console.log(audio_file);

function load_notes(notes_arr) {
    let res = [];
    for(let note of notes_arr) {
        res.push(JSON.parse(note));
    }
    return res;
}

const notes = load_notes(notesArr);
console.log(notes);

var play_button = window.document.getElementById('play_button');
//play_button.addEventListener('click', playAudio)

var stop_button = window.document.getElementById('stop_button');
//stop_button.addEventListener('click', stopAudio)

var reset_button = window.document.getElementById('reset_button');
//reset_button.addEventListener('click', reset)

const timer = new Timer(INTERVAL, notes, audio_file);

function start() {
    timer.start();
}

function pause() {
    timer.pause();
}

function reset() {
    timer.reset();
}

play_button.addEventListener('click', start);
stop_button.addEventListener('click', pause);
reset_button.addEventListener('click', reset);