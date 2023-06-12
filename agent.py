class Agent:
  """
  inner_lb, inner_ub: inner window boundaries in number of frames
  outer_lb, outer_ub: outer window boundaries in number of frames
  taken from: https://www.semanticscholar.org/paper/Evaluating-the-Online-Capabilities-of-Onset-Methods-B%C3%B6ck-Krebs/f2696e2fb526f19a0f67e286cb0d8205bc30f8e9
  """
  def __init__(self, initial_event, t, tempo_hypothesis, inner_lb, inner_ub, outer_lb, outer_ub):
    self.initial_event = initial_event
    self.initial_tempo = t
    self.tempo_hypothesis = tempo_hypothesis
    self.inner_ub = inner_ub
    self.inner_lb = inner_lb
    self.outer_ub = outer_ub
    self.outer_lb = outer_lb
    self.update_factor = 0.25

    self.detetected_beats = [initial_event]

  """
  check if event is in inner and/or outer window
  """
  def process_event(self, event):
    # add new event to beats if last detected beat + tempo hypothesis (phase) is in window
    last_beat = self.detetected_beats[-1]
    beat_prediction = last_beat + self.tempo_hypothesis

    in_inner_window = False
    if beat_prediction - self.inner_lb <= event <= beat_prediction + self.inner_ub:
      in_inner_window = True

      # add event to detected beats
      self.detetected_beats.append(event)

      # update tempo hypothesis by a factor times the diff of prediction and actual beat
      # case 1: event before prediction -> tempo is faster than predicted, therefore increase tempo hypothesis 
      # case 2: event after prediction -> tempo is slower than predicted, therefore decrease tempo hypothesis 
      self.tempo_hypothesis += self.update_factor * (beat_prediction - event)

    # interpolate if last beat + tempo hypothesis (phase) is smaller than beat prediction
    # TODO insert interpolated beats

    in_outer_window = False 
    if beat_prediction - self.outer_lb <= event <= beat_prediction + self.outer_ub:
      in_outer_window = True