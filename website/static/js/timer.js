class Timer {

    constructor(interval, notes, audio_file) {
        this.interval = interval;
        this.notes = notes;
        this.idx = 0;
        this.counter = 0;
        this.isRunning = false;
        this.timeouts = [];
        this.audio_file = audio_file;
        this.backlog = [];
        this.audio_file.addEventListener('ended', () => {
            this.reset();
        })
    }

    start() {
        this.isRunning = true;
        this.initialize();
        this.audio_file.play();
        this.mainLoop();
    }

    clearTimeouts() {
        for(let i = 0; i < this.timeouts.length; i++) {
            clearTimeout(this.timeouts[i][0]);
        }
    }

    resetKeys() {
        let keys = window.document.querySelectorAll('.key');
        for(let i = 0; i < keys.length; i++) {
            keys[i].classList.remove('active');
        }
    }

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

    pause() {
        this.isRunning = false;
        this.audio_file.pause();
        this.backlog = [];
        let curTime = Date.now();
        for(let i = 0; i < this.timeouts.length; i++) {
            let curTimeout = this.timeouts[i];
            let expected = curTimeout[1] + curTimeout[2] * 1000;
            let idx = curTimeout[3];
            console.log("expected: " + expected + " curTime: " + curTime);
            if (curTime <= expected) {
                let remaining = expected - curTime;
                console.log("remaining: " + remaining);
                this.backlog.push([remaining, idx]);
            }
        }
        this.clearTimeouts();
        this.timeouts = [];
    }


    initialize() {
        for(let i = 0; i < this.backlog.length; i++) {
            let timeRemaining = this.backlog[i][0];
            let idx = this.backlog[i][1];
            let timeoutID = setTimeout(() => {
                this.removeClick(idx);
            }, timeRemaining);
            this.timeouts.push([timeoutID, Date.now(), timeRemaining, idx]);
        }
    }

    removeClick(idx) {
        if (idx >= this.notes.length | !this.isRunning) {
            return;
        }
        let note = this.notes[idx];
        let name = note.name;
        console.log("remove " + this.counter + " " + name);
        let key = window.document.getElementById(name);
        key.classList.remove('active');
    }

    addClick(name) {
        if (!this.isRunning) {
            return;
        }
        let key = window.document.getElementById(name);
        console.log("add " + this.counter + " " + name);
        key.classList.add('active');
    }


    mainLoop() {
        if (!this.isRunning) {
            return;
        }
        if (this.idx >= this.notes.length) {
            this.done = true;
            this.reset();
        }
        let curNote = this.notes[this.idx];
        let timeStep = curNote.time_step;
        let duration = curNote.duration;
        let start = Date.now();
        if (this.counter >= timeStep) {
            this.addClick(curNote.name);
            let idx = this.idx;
            let curSleep = duration * 1000;
            console.log(curSleep);
            let timeoutID = setTimeout(() => {
                this.removeClick(idx);
            }, curSleep);
            this.timeouts.push([timeoutID, Date.now(), duration, this.idx]);
            this.idx++;
        }
        this.counter += this.interval / 1000;
        let after = Date.now();
        let sleep = (this.interval - (after - start));
        console.log(sleep);
        setTimeout(() => {
            this.mainLoop();
        }, this.interval - (after - start))

    }

}