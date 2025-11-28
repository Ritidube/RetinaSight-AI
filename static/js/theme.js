// // Theme toggle
// const btns = document.querySelectorAll('#themeBtn');

// function toggleTheme(){
//   document.body.classList.toggle('dark');
// }

// btns.forEach(b => b.addEventListener('click', toggleTheme));


// // Live clock
// function startClock(){
//   const el = document.getElementById('clock');
//   if(!el) return;

//   function tick(){
//     const now = new Date();
//     el.textContent = now.toLocaleString();
//   }
//   tick();
//   setInterval(tick, 1000);
// }

// startClock();


// document.getElementById("themeBtn").addEventListener("click", () => {
//     document.body.classList.toggle("dark");
// });

// // Live Clock
// function updateClock() {
//     const now = new Date();
//     document.getElementById("clock").innerText = now.toLocaleString();
// }
// setInterval(updateClock, 1000);
// updateClock();


document.addEventListener("DOMContentLoaded", () => {
    let savedTheme = localStorage.getItem("theme") || "dark";

    if (savedTheme === "light") {
        document.body.classList.add("light");
    }

    document.getElementById("themeToggle").onclick = () => {
        document.body.classList.toggle("light");

        let theme = document.body.classList.contains("light") ? "light" : "dark";
        localStorage.setItem("theme", theme);
    };
});
