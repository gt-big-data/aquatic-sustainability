/* ===== TEAM DATA ===== */

const teamLeads = [
  {
    name: "Rick Lee",
    username: "rlee06",
    role: "Platform & Data Viz Lead"
  },
  {
    name: "Michael Xu",
    username: "mykelxu",
    role: "Analysis Lead"
  }
];

//TODO: Update member info here
const teamMembers = [
  { name: "William Gao", username: "william-gao-709a0a17b", role: "Platform" },
  { name: "Joshua Hevelow", username: "joshhevelow", role: "Data Viz" },
  { name: "Quinten Lemoine", username: "quintenlemoine", role: "Analysis" },
  { name: "Surya Narasimhan", username: "surya-narasimhan", role: "Analysis" },
  { name: "Anish Neema", username: "anishneema", role: "Analysis" },
  { name: "Deepesh Raj", username: "deepesh-raj", role: "Data Viz" },
  { name: "Anoushka Scaria", username: "anoushka-scaria-0a8080313", role: "Data Viz" },
  { name: "Grace Shao", username: "grace-shao-gt-cs", role: "Platform" },
  { name: "Vaibhav Wudaru", username: "vaibhav-wudaru-9605aa248", role: "Platform/Analysis" }
  
];

/* ===== CARD GENERATOR ===== */

function createCard(member, isLead = false) {
  const card = document.createElement("div");
  card.className = "team-card" + (isLead ? " lead" : "");

  const avatar = document.createElement("img");
  avatar.className = "team-avatar";

  // Fallback avatar (safe)
    avatar.src = `/static/assets/images/team/${member.username}.jpg`;
    avatar.onerror = () => {
    avatar.src = `https://ui-avatars.com/api/?name=${encodeURIComponent(member.name)}`;
    };

  const name = document.createElement("h3");
  name.className = "team-name";
  name.textContent = member.name;

  const link = document.createElement("a");
  link.className = "team-link";
  link.href = `https://linkedin.com/in/${member.username}`;
  link.target = "_blank";
  link.textContent = "LinkedIn";

  card.appendChild(avatar);
  card.appendChild(name);


    const role = document.createElement("p");
    role.className = "team-role";
    role.textContent = member.role;
    card.appendChild(role);

  card.appendChild(link);

  return card;
}

/* ===== RENDER ===== */

const leadsContainer = document.getElementById("teamLeads");
const membersContainer = document.getElementById("teamGrid");

teamLeads.forEach(member => {
  leadsContainer.appendChild(createCard(member, true));
});

teamMembers.forEach(member => {
  membersContainer.appendChild(createCard(member));
});