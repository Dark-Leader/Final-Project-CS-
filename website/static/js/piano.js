const INTERVAL = 250; // 250ms -> duration of shortest note.

const audio_file = document.getElementById('audio_file'); // get output audio file
audio_file.load();

function load_notes(notes_arr) {
    /**
     * construct notes from JSON strings of Note objects.
     * @type {*[JSON]} notes_arr
     */
    let res = [];
    for(let note of notes_arr) {
        res.push(JSON.parse(note));
    }
    return res;
}

const notes = load_notes(notesArr); // load array of notes.

// get buttons
const play_button = window.document.getElementById('play_button');
const stop_button = window.document.getElementById('stop_button');
const reset_button = window.document.getElementById('reset_button');

// create timer
const timer = new Timer(INTERVAL, notes, audio_file);

/**
 * start playing the melody.
 */
function start() {
    timer.start();
}

/**
 * pause the melody.
 */
function pause() {
    timer.pause();
}

/**
 * reset melody to beginning.
 */
function reset() {
    timer.reset();
}

// set event listeners
play_button.addEventListener('click', start);
stop_button.addEventListener('click', pause);
reset_button.addEventListener('click', reset);