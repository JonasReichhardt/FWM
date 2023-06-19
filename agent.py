import copy as cp

class Agent:
  """
  inner_lb, inner_ub: inner window boundaries in number of frames
  outer_lb, outer_ub: outer window boundaries in number of frames
  taken from: https://www.semanticscholar.org/paper/Evaluating-the-Online-Capabilities-of-Onset-Methods-B%C3%B6ck-Krebs/f2696e2fb526f19a0f67e286cb0d8205bc30f8e9
  """
  def __init__(self, initial_event, initial_tempo, tempo_hypothesis, inner_lb, inner_ub, outer_lb, outer_ub, onsets, onset_energy, score, beats, min_tempo):
    self.initial_event = initial_event
    self.initial_tempo = initial_tempo
    self.tempo_hypothesis = tempo_hypothesis
    self.inner_ub = inner_ub
    self.inner_lb = inner_lb
    self.outer_ub = outer_ub
    self.outer_lb = outer_lb
    self.score = score
    self.onsets = onsets
    self.onset_energy = onset_energy
    self.min_tempo = min_tempo
    self.est_mult = 1
    self.update_factor = 0.25

    if len(beats) == 0:
      self.beats = [(initial_event, True)]
    else:
      self.beats = cp.deepcopy(beats)

  def __eq__(self, other): 
      if not isinstance(other, Agent):
          return False

      return self.initial_event == other.initial_event and self.initial_tempo == other.initial_tempo

  """
  check if event is in inner and/or outer window
  """
  def process(self, onset_idx):
    beat_prediction = self.beats[-1][0]
    new_agents = []

    while beat_prediction < onset_idx:
      last_beat = self.beats[-1][0]
      # add new event to beats if last detected beat + tempo hypothesis (phase) is in window
      beat_prediction = last_beat + (self.tempo_hypothesis * self.est_mult)

      # beat candidate is beat with the highest energy inside the window
      try:
        beat_candidate = self.get_beat_candidate(beat_prediction)

        in_inner_window = beat_prediction - self.inner_lb <= beat_candidate <= beat_prediction + self.inner_ub
        in_outer_window = beat_prediction - self.outer_lb <= beat_candidate <= beat_prediction + self.outer_ub 

        error = abs(beat_prediction - beat_candidate) / (self.outer_lb + self.outer_ub)
        normalization_factor = self.tempo_hypothesis/ self.min_tempo / 70

        if in_inner_window or in_outer_window:
          # interpolate beats based on current tempo hypothesis
          frame_diff = beat_candidate - last_beat

          if frame_diff > self.tempo_hypothesis:
            interpolate_beats = int(frame_diff/self.tempo_hypothesis)
            dist = frame_diff/(interpolate_beats + 1)

            for i in range(1, interpolate_beats):
              self.beats.append((last_beat + dist * i, False)) # type: ignore

          # add event to detected beats
          self.beats.append((beat_candidate, True))

          if in_inner_window:
            # increase agent score
            self.score += 1 - error * self.onset_energy[beat_candidate] * normalization_factor
          else:
            # decrease agent score
            self.score -= 2 * error * self.onset_energy[beat_candidate] * normalization_factor

            # create new agent if in_outer_window is true
            new_agents.append(Agent(beat_candidate, self.tempo_hypothesis, self.tempo_hypothesis, 
                        self.inner_lb, self.inner_ub, self.outer_lb, self.outer_ub, 
                        self.onsets, self.onset_energy, self.score * 0.9, self.beats, self.min_tempo))
        else:
          # decrease agent score
          self.score -= 4 * error * self.onset_energy[beat_candidate] * normalization_factor
        
        self.est_mult = 1

        self.update_tempo_hypothesis(beat_prediction, beat_candidate)
      except ValueError:
        self.est_mult += 1

    return new_agents

  """
  update tempo hypothesis by a factor times the diff of prediction and actual beat
  case 1: event before prediction -> tempo is faster than predicted, therefore increase tempo hypothesis 
  case 2: event after prediction -> tempo is slower than predicted, therefore decrease tempo hypothesis
  """
  def update_tempo_hypothesis(self, beat_prediction, event_frame):
      self.tempo_hypothesis += self.update_factor * (beat_prediction - event_frame) / 2
      outer_window = int(self.tempo_hypothesis * 0.45)
      self.outer_lb = outer_window
      self.outer_ub = outer_window
  
  """
  get highest onset in inside the outer and inner window
  """
  def get_beat_candidate(self,beat_prediction):
    lb = beat_prediction - self.outer_lb
    ub = beat_prediction + self.outer_ub

    candidate_energy = dict()
    for c in list(filter(lambda n: n > lb and n < ub, self.onsets)):
      candidate_energy[c] = self.onset_energy[c]
    return max(candidate_energy, key=candidate_energy.get) # type: ignore


