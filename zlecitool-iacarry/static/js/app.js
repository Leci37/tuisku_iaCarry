/* La trastienda de iaCarry: lo poco que hace falta en el navegador.

   * Una foto que ya ha borrado el borrado por antigüedad no deja una imagen
     rota en «Compras»: dice «borrada».
   * En «Marca», el color elegido se ve en la vista previa antes de guardar
     (la paleta de verdad la saca el servidor al guardar). */
(function () {
  "use strict";
  document.addEventListener("error", function (event) {
    var img = event.target;
    if (!img || img.tagName !== "IMG" || !img.classList.contains("ia-thumb")) return;
    var label = img.parentNode.querySelector("[data-i18n='iaPhotoGone']");
    img.remove();
    if (label) label.hidden = false;
  }, true);

  var color = document.querySelector("input[name='color'].ia-color-big");
  var preview = document.querySelector(".ia-brand-preview");
  if (color && preview) {
    color.addEventListener("input", function () {
      preview.style.setProperty("--p-pri", color.value);
      preview.style.setProperty("--p-ink", color.value);
    });
  }
})();
