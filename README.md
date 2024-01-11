# spectrogram_TF

Unsupervised adversarial domain adaptation for spectrogram images (`simulation` to `reality` transfer learning).

<table>
<thead>
  <tr>
    <th></th>
    <th>Source Only</th>
    <th>Domain Adptation</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Test Accuracy</td>
    <td>90.8%</td>
    <td>94.1%</td>
  </tr>
  <tr>
    <td>Confusion Chart</td>
    <td><img src = "./output_source/Test_ConfMatrix.png"></img></td>
    <td><img src = "./output_DA/Test_ConfMatrix.png"></img</td>
  </tr>
  <tr>
    <td>TSNE for test</td>
    <td><img src = "./output_source/Test_TSNE.png"></td>
    <td><img src = "./output_DA/Test_TSNE.png"></td>
  </tr>
  <tr>
    <td>TSNE for all</td>
    <td><img src = "./output_source/ALL_TSNE.png"></td>
    <td><img src = "./output_DA_2/ALL_TSNE.png"></td>
  </tr>
</tbody>
</table>