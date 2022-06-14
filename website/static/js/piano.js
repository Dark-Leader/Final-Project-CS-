const keys = document.getElementById("piano").children;
console.log(keys)

function load_notes(notes) {
    var res = [];
    for(var note of notes) {
        res.push(JSON.parse(note));
    }
    return res;
}
console.log(notesArr);
var notes = load_notes(notesArr);
console.log(notes);

for (var n of notes) {
    console.log(n.pitch);
}