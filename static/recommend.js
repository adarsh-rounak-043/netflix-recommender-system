$(function() {
  const source = document.getElementById('autoComplete');
  const inputHandler = function(e) {
    $('.movie-button').attr('disabled', e.target.value === "");
  };
  source.addEventListener('input', inputHandler);

  $('.movie-button').on('click', function() {
    const title = $('.movie').val();
    if (title === "") {
      $('.results').hide();
      $('.fail').show();
    } else {
      load_details(api_key, title);
    }
  });
});

function recommendcard(e) {
  const title = e.getAttribute('title');
  load_details(api_key, title);
}

function load_details(my_api_key, title) {
  $.ajax({
    type: 'GET',
    url: `https://api.themoviedb.org/3/search/movie?api_key=${my_api_key}&query=${title}`,
    success: function(movie) {
      if (movie.results.length < 1) {
        $('.fail').show();
        $('.results').hide();
        $("#loader").delay(500).fadeOut();
      } else {
        $("#loader").fadeIn();
        $('.fail').hide();
        $('.results').delay(1000).show();
        const movie_id = movie.results[0].id;
        const movie_title = movie.results[0].original_title;
        movie_recs(movie_title, movie_id, my_api_key);
      }
    },
    error: function() {
      alert('Invalid Request');
      $("#loader").delay(500).fadeOut();
    },
  });
}

function movie_recs(movie_title, movie_id, my_api_key) {
  $.ajax({
    type: 'POST',
    url: "/similarity",
    data: { name: movie_title },
    success: function(recs) {
      if (recs.startsWith("Sorry!")) {
        $('.fail').show();
        $('.results').hide();
        $("#loader").delay(500).fadeOut();
      } else {
        $('.fail').hide();
        $('.results').show();
        const arr = recs.split('---');
        get_movie_details(movie_id, my_api_key, arr, movie_title);
      }
    },
    error: function(e) {
      console.log(e);
      alert("Error fetching recommendations");
      $("#loader").delay(500).fadeOut();
    },
  });
}

function get_movie_details(movie_id, my_api_key, arr, movie_title) {
  $.ajax({
    type: 'GET',
    url: `https://api.themoviedb.org/3/movie/${movie_id}?api_key=${my_api_key}`,
    success: function(movie_details) {
      show_details(movie_details, arr, movie_title, my_api_key, movie_id);
    },
    error: function() {
      alert("API Error!");
      $("#loader").delay(500).fadeOut();
    },
  });
}

function show_details(movie_details, arr, movie_title, my_api_key, movie_id) {
  const imdb_id = movie_details.imdb_id;
  const poster = `https://image.tmdb.org/t/p/original${movie_details.poster_path}`;
  const overview = movie_details.overview;
  const genres = movie_details.genres.map(g => g.name).join(", ");
  const rating = movie_details.vote_average;
  const vote_count = movie_details.vote_count.toLocaleString();
  const release_date = new Date(movie_details.release_date).toDateString().split(' ').slice(1).join(' ');
  let runtime = parseInt(movie_details.runtime);
  runtime = `${Math.floor(runtime / 60)} hour(s) ${runtime % 60} min(s)`;
  const status = movie_details.status;

  const arr_posters = get_movie_posters(arr, my_api_key);
  const movie_cast = get_movie_cast(movie_id, my_api_key);
  const ind_cast = get_individual_cast(movie_cast, my_api_key);

  const details = {
    title: movie_title,
    cast_ids: JSON.stringify(movie_cast.cast_ids),
    cast_names: JSON.stringify(movie_cast.cast_names),
    cast_chars: JSON.stringify(movie_cast.cast_chars),
    cast_profiles: JSON.stringify(movie_cast.cast_profiles),
    cast_bdays: JSON.stringify(ind_cast.cast_bdays),
    cast_bios: JSON.stringify(ind_cast.cast_bios),
    cast_places: JSON.stringify(ind_cast.cast_places),
    imdb_id: imdb_id,
    poster: poster,
    genres: genres,
    overview: overview,
    rating: rating,
    vote_count: vote_count,
    release_date: release_date,
    runtime: runtime,
    status: status,
    rec_movies: JSON.stringify(arr),
    rec_posters: JSON.stringify(arr_posters),
  };

  $.ajax({
    type: 'POST',
    data: details,
    url: "/recommend",
    dataType: 'html',
    complete: function() {
      $("#loader").delay(500).fadeOut();
    },
    success: function(response) {
      $('.results').html(response);
      $('#autoComplete').val('');
      $(window).scrollTop(0);
    }
  });
}

function get_individual_cast(movie_cast, my_api_key) {
  const cast_bdays = [], cast_bios = [], cast_places = [];

  for (const cast_id of movie_cast.cast_ids) {
    $.ajax({
      type: 'GET',
      url: `https://api.themoviedb.org/3/person/${cast_id}?api_key=${my_api_key}`,
      async: false,
      success: function(details) {
        cast_bdays.push(new Date(details.birthday).toDateString().split(' ').slice(1).join(' '));
        cast_bios.push(details.biography);
        cast_places.push(details.place_of_birth);
      }
    });
  }

  return { cast_bdays, cast_bios, cast_places };
}

function get_movie_cast(movie_id, my_api_key) {
  const cast_ids = [], cast_names = [], cast_chars = [], cast_profiles = [];

  $.ajax({
    type: 'GET',
    url: `https://api.themoviedb.org/3/movie/${movie_id}/credits?api_key=${my_api_key}`,
    async: false,
    success: function(res) {
      const top_cast = res.cast.slice(0, 10);
      for (const cast of top_cast) {
        cast_ids.push(cast.id);
        cast_names.push(cast.name);
        cast_chars.push(cast.character);
        cast_profiles.push(`https://image.tmdb.org/t/p/original${cast.profile_path}`);
      }
    },
    error: function() {
      alert("Invalid Request!");
      $("#loader").delay(500).fadeOut();
    }
  });

  return { cast_ids, cast_names, cast_chars, cast_profiles };
}

function get_movie_posters(arr, my_api_key) {
  const posters = [];

  for (const title of arr) {
    $.ajax({
      type: 'GET',
      url: `https://api.themoviedb.org/3/search/movie?api_key=${my_api_key}&query=${title}`,
      async: false,
      success: function(res) {
        posters.push(`https://image.tmdb.org/t/p/original${res.results[0].poster_path}`);
      },
      error: function() {
        alert("Invalid Request!");
        $("#loader").delay(500).fadeOut();
      },
    });
  }

  return posters;
}
