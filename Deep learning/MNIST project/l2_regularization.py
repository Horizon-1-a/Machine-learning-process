def regularize():
  t1 = np.diag(np.full(256, lamda))
  t2 = np.diag(np.full(10, lamda))
  #forward pass {}
 #backpropagation
  w2_r = np.hstack((w2[:, :-1], np.zeros((10, 1))))
  w1_r = np.hstack((w1[:, : -1], np.zeros((256, 1))))
  dz2 = 1 / batch_size * (a2 - y_cur)
  dw2 = dz2 @ a1_bar.T + 1 / batch_size * (t2.T @ (t2 @ w2_r))
  da1 = w2[:, :-1].T @ dz2
  dz1 = d_relu(z1) * da1
  dw1 = dz1 @ x_cur.T + (t1.T @ (t1 @ w1_r)) * 1 / batch_size
