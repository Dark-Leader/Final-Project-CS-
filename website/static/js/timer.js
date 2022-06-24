/**
 * Represents a timer that executes notes according to their timeStep and updates the display.
 */

class Timer {

    /**
     * constructor.
     * @param {int} interval - min duration of a note.
     * @param {array[JSON]} notes - array of notes in the melody.
     * @param {Audio} audio_file - audio file of the melody.
     */
    constructor(interval, notes, audio_file) {
        this.interval = interval;
        this.notes = notes;
        this.idx = 0;
        this.counter = 0;
        this.isRunning = false;
        this.timeouts = [];
        this.audio_file = audio_file;
        this.backlog = [];
    }

    /**
     * starts playing the melody and animation of the piano.
     */
    start() {
        if (this.isRunning) {
            return;
        }
        this.isRunning = true;
        this.initialize();
        this.audio_file.play();
        this.mainLoop();
    }

    /**
     * clears all pending timeouts.
     */
    clearTimeouts() {
        for(let i = 0; i < this.timeouts.length; i++) {
            clearTimeout(this.timeouts[i][0]);
        }
    }

    /**
     * resets all keys to not active state.
     */
    resetKeys() {
        let keys = window.document.querySelectorAll('.key');
        for(let i = 0; i < keys.length; i++) {
            keys[i].classList.remove('active');
        }
    }

    getLongestWait() {
        let max = 0;
        let curTime = Date.now();
        for (let i = 0; i < this.timeouts.length; i++) {
            let curTimeout = this.timeouts[i];
            let expected = curTimeout[1] + curTimeout[2] * 1000;
            if (expected - curTime > max) {
                max = expected - curTime;
            }
        }
        return max;
    }


    /**
     * resets the audio file to the beginning, removes all pending timeouts, resets the piano keys to not active state.
     */
    reset() {
        this.clearTimeouts();
        this.isRunning = false;
        this.resetKeys();
        this.timeouts = [];
        this.backlog = [];
        this.idx = 0;
        this.counter = 0;
        this.audio_file.pause();
        this.audio_file.currentTime = 0;
    }

    /**
     * pauses the melody playback, saving pending timeouts.
     */
    pause() {
        if (!this.isRunning) {
            return;
        }
        this.isRunning = false;
        this.audio_file.pause();
        this.backlog = [];
        let curTime = Date.now();
        for(let i = 0; i < this.timeouts.length; i++) {
            let curTimeout = this.timeouts[i];
            let expected = curTimeout[1] + curTimeout[2] * 1000;
            let idx = curTimeout[3];
            if (curTime <= expected) { // if this is a pending timeout -> didn't happen yet.
                let remaining = expected - curTime;
                this.backlog.push([remaining, idx]); // add timeout to backlog.
            }
        }
        this.clearTimeouts();
        this.timeouts = [];
    }

    /**
     * initialize the backlog -> if there were pending timeouts we need to recreate them with updated remaining time
     * to be executed.
     */
    initialize() {
        for(let i = 0; i < this.backlog.length; i++) {
            let timeRemaining = this.backlog[i][0];
            let idx = this.backlog[i][1];
            let timeoutID = setTimeout(() => { // create timeout to set key to not active state.
                this.removeClick(idx);
            }, timeRemaining);
            this.timeouts.push([timeoutID, Date.now(), timeRemaining, idx]);
        }
    }

    /**
     * sets piano key to not active state.
     * @param {int} idx - index of note in melody who needs to be updated to not active state.
     */
    removeClick(idx) {
        if (idx >= this.notes.length | !this.isRunning) {
            return;
        }
        let note = this.notes[idx];
        let name = note.name;
        let key = window.document.getElementById(name);
        if (key != null) {
            key.classList.remove('active');
        }
    }

    /**
     * sets piano key to active state.
     * @param {string} name - name of piano key who needs to be updated to active state.
     */
    addClick(name) {
        if (!this.isRunning) {
            return;
        }
        let key = window.document.getElementById(name);
        if (key != null) {
            key.classList.add('active');
        }
    }

    /**
     * mainloop of the playback.
     * at each interval we check if current note needs to be played,
     * if a not needs to be played - we play it and set a timeout to set
     * it back to not active state.
     */
    mainLoop() {
        if (!this.isRunning) {
            return;
        }
        if (this.idx >= this.notes.length) {
            let wait = this.getLongestWait();
            setTimeout(() => {
                this.reset();
            }, wait); // call mainloop again in 'updatedDelay' milliseconds.
            return;
        }
        let curNote = this.notes[this.idx];
        let timeStep = curNote.time_step;
        let duration = curNote.duration;
        let start = Date.now();
        if (this.counter >= timeStep) { // need to play current note.
            this.addClick(curNote.name);
            let idx = this.idx;
            let delay = duration * 1000;
            let timeoutID = setTimeout(() => { // set timeout to set key to not active state in 'delay' milliseconds.
                this.removeClick(idx);
            }, delay);
            this.timeouts.push([timeoutID, Date.now(), duration, this.idx]);
            this.idx++;
        }
        this.counter += this.interval / 1000;
        let after = Date.now();
        let updatedDelay = (this.interval - (after - start));
        setTimeout(() => {
            this.mainLoop();
        }, updatedDelay); // call mainloop again in 'updatedDelay' milliseconds.

    }

}