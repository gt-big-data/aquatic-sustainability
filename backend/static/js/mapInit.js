/**
 * Shared Google Maps initialization for all pages
 */
async function loadGoogleMaps() {
  try {
    const response = await fetch("http://127.0.0.1:8080/api/config/maps-key");
    const data = await response.json();
    const apiKey = data.googleMapsApiKey;

    if (!apiKey) {
      throw new Error("Missing Google Maps API key");
    }

    await new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = `https://maps.googleapis.com/maps/api/js?key=${apiKey}&v=weekly&libraries=maps`;
      script.async = true;
      script.defer = true;
      script.onload = resolve;
      script.onerror = reject;
      document.head.appendChild(script);
    });

    const { Map } = await google.maps.importLibrary("maps");
    const map = new Map(document.getElementById("map"), {
      center: { lat: 33.749, lng: -84.388 },
      zoom: 8,
      mapTypeId: "terrain",
    });

  } catch (error) {
    console.error("Error initializing Google Maps:", error);
  }
}

loadGoogleMaps();
